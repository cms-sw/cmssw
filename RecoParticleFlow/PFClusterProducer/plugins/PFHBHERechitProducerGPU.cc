#include <array>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

// Comment out to disable debugging
//#define PF_DEBUG_ENABLE

#ifdef PF_DEBUG_ENABLE
#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>
#endif  // PF_DEBUG_ENABLE

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

#include "DeclsForKernels.h"
#include "SimplePFGPUAlgos.h"

typedef PFHCALDenseIdNavigator<HcalDetId, HcalTopology, false> PFRecHitHCALDenseIdNavigator;

class PFHBHERechitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PFHBHERechitProducerGPU(edm::ParameterSet const&);
  ~PFHBHERechitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  unsigned getIdx(const unsigned);

  //Output Product Type
  using PFRecHitSoAProductType = cms::cuda::Product<PFRecHit::HCAL::OutputPFRecHitDataGPU>;
  //Output Token
  using IProductType = cms::cuda::Product<hcal::RecHitCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<IProductType> InputRecHitSoA_Token_;

  using OProductType = cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>;
  edm::EDPutTokenT<OProductType> OutputPFRecHitSoA_Token_;

  PFRecHit::HCAL::OutputPFRecHitDataGPU outputGPU;
  cms::cuda::ContextState cudaState_;

  hcal::PFRecHitCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFRecHits;

  std::unique_ptr<PFRecHitNavigatorBase> navigator_;

  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESHandle<CaloGeometry> geoHandle;
  edm::ESHandle<HcalTopology> topoHandle;

  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_;
  std::unique_ptr<const HcalTopology> topology_;

  PFRecHit::HCAL::PersistentDataCPU persistentDataCPU;
  PFRecHit::HCAL::PersistentDataGPU persistentDataGPU;
  PFRecHit::HCAL::ScratchDataGPU scratchDataGPU;

  uint32_t nValidBarrelIds = 0, nValidEndcapIds = 0, nValidDetIds = 0;
  float qTestThresh = 0.;
  std::vector<std::vector<DetId>>* neighboursHcal_;
  std::vector<unsigned>* vDenseIdHcal;
  std::unordered_map<unsigned, unsigned>
      detIdToIndex;  // Mapping of detId to index. Use this index instead of raw detId to encode neighbours
  std::vector<GlobalPoint> validDetIdPositions;
  unsigned denseIdHcalMax_ = 0;
  unsigned denseIdHcalMin_ = 0;

  bool initCuda = true;
  uint32_t nPFRHTotal = 0;
  std::array<float, 5> GPU_timers;

#ifdef PF_DEBUG_ENABLE
  TTree* tree;
  TFile* f;
  TH1F* hTimers = new TH1F("timers", "GPU kernel timers ", 5, -0.5, 4.5);

  Int_t numEvents = 0;
#endif
};

PFHBHERechitProducerGPU::PFHBHERechitProducerGPU(edm::ParameterSet const& ps)
    : InputRecHitSoA_Token_{consumes<IProductType>(
          ps.getParameterSetVector("producers")[0].getParameter<edm::InputTag>("src"))},
      OutputPFRecHitSoA_Token_{produces<OProductType>(ps.getParameter<std::string>("PFRecHitsGPUOut"))},
      hcalToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      geomToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()) {
  edm::ConsumesCollector cc = consumesCollector();

  produces<reco::PFRecHitCollection>();
  produces<reco::PFRecHitCollection>("Cleaned");

  const auto& prodConf = ps.getParameterSetVector("producers")[0];
  const std::string& prodName = prodConf.getParameter<std::string>("name");
  const auto& qualityConf = prodConf.getParameterSetVector("qualityTests");

  const std::string& qualityTestName = qualityConf[0].getParameter<std::string>("name");
  qTestThresh = (float)qualityConf[0].getParameter<double>("threshold");

  const auto& navSet = ps.getParameterSet("navigator");
  navigator_ = PFRecHitNavigationFactory::get()->create(navSet.getParameter<std::string>("name"), navSet, cc);

#ifdef PF_DEBUG_ENABLE
  tree = new TTree("tree", "tree");
  tree->Branch("Event", &numEvents);
  tree->Branch("timers", &GPU_timers);

  hTimers->GetYaxis()->SetTitle("time (ms)");
  hTimers->GetXaxis()->SetBinLabel(1, "initializeArrays");
  hTimers->GetXaxis()->SetBinLabel(2, "buildDetIdMap");
  hTimers->GetXaxis()->SetBinLabel(3, "applyQTests");
  hTimers->GetXaxis()->SetBinLabel(4, "applyMask");
  hTimers->GetXaxis()->SetBinLabel(5, "convertToPFRecHits");
#endif
}

PFHBHERechitProducerGPU::~PFHBHERechitProducerGPU() {
  topology_.release();
#ifdef PF_DEBUG_ENABLE
  TFile* f = new TFile("gpuPFRecHitTimers.root", "recreate");
  f->cd();
  tree->Write();
  hTimers->Scale(1. / numEvents);
  hTimers->Write();
  delete f;
#endif
}

void PFHBHERechitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;

  //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerGPU"});
  //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hltHbherecoGPU"});

  desc.add<std::string>("PFRecHitsGPUOut", "");
  // Prevents the producer and navigator parameter sets from throwing an exception
  // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
  desc.setAllowAnything();

  cdesc.addWithDefaultLabel(desc);
}

unsigned PFHBHERechitProducerGPU::getIdx(const unsigned denseid) { return (denseid - denseIdHcalMin_); }

void PFHBHERechitProducerGPU::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  navigator_->init(setup);
  if (!theRecNumberWatcher_.check(setup))
    return;

  topoHandle = setup.getHandle(hcalToken_);
  topology_.release();
  topology_.reset(topoHandle.product());

  // Get list of valid Det Ids for HCAL barrel & endcap once
  geoHandle = setup.getHandle(geomToken_);
  // get the hcal geometry
  const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  const std::vector<DetId>& validBarrelDetIds = hcalBarrelGeo->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& validEndcapDetIds = hcalEndcapGeo->getValidDetIds(DetId::Hcal, HcalEndcap);
  nValidBarrelIds = validBarrelDetIds.size();
  nValidEndcapIds = validEndcapDetIds.size();
  nValidDetIds = nValidBarrelIds + nValidEndcapIds;

  std::cout << "Found nValidBarrelIds = " << nValidBarrelIds << "\tnValidEndcapIds = " << nValidEndcapIds << std::endl;

  detIdToIndex.clear();
  validDetIdPositions.clear();

  detIdToIndex.reserve(nValidDetIds);
  validDetIdPositions.reserve(nValidDetIds);
  //vDenseIdHcal.clear();
  //vDenseIdHcal.reserve(nValidDetIds);

  //    for (auto hDetId : validBarrelDetIds) {
  //      vDenseIdHcal.push_back(topology_.get()->detId2denseId(hDetId));
  //    }
  //    for (auto hDetId : validEndcapDetIds) {
  //      vDenseIdHcal.push_back(topology_.get()->detId2denseId(hDetId));
  //    }
  //    std::sort(vDenseIdHcal.begin(), vDenseIdHcal.end());
  //    neighboursHcal_ = reinterpret_cast<PFRecHitHCALDenseIdNavigator*>(*(&navigator_))->getNeighbours();
  //    std::cout<<"Found neighboursHcal_->.size() = "<<neighboursHcal_->size()<<std::endl;
  vDenseIdHcal = reinterpret_cast<PFRecHitHCALDenseIdNavigator*>(&(*navigator_))->getValidDenseIds();
  std::cout << "Found vDenseIdHcal->size() = " << vDenseIdHcal->size() << std::endl;

  for (const auto& denseid : *vDenseIdHcal) {
    DetId detid_c = topology_.get()->denseId2detId(denseid);
    HcalDetId hid_c = HcalDetId(detid_c);

    if (hid_c.subdet() == HcalBarrel)
      validDetIdPositions.emplace_back(hcalBarrelGeo->getGeometry(detid_c)->getPosition());
    else if (hid_c.subdet() == HcalEndcap)
      validDetIdPositions.emplace_back(hcalEndcapGeo->getGeometry(detid_c)->getPosition());
    else
      std::cout << "Invalid subdetector found for detId " << hid_c.rawId() << ": " << hid_c.subdet() << std::endl;

    detIdToIndex[hid_c.rawId()] = detIdToIndex.size();
  }

  initCuda = true;  // (Re)initialize cuda arrays
}

void PFHBHERechitProducerGPU::acquire(edm::Event const& event,
                                      edm::EventSetup const& setup,
                                      edm::WaitingTaskWithArenaHolder holder) {
  GPU_timers.fill(0.0);
  //auto start = std::chrono::high_resolution_clock::now();

  auto const& HBHERecHitSoAProduct = event.get(InputRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{HBHERecHitSoAProduct, std::move(holder), cudaState_};
  auto const& HBHERecHitSoA = ctx.get(HBHERecHitSoAProduct);
  size_t num_rechits = HBHERecHitSoA.size;
  //std::cout << "num input rechits = " << num_rechits << "\tctx.stream() = " << ctx.stream() << std::endl;

  // Lambda function to copy arrays to CPU for testing
  //    auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
  //        using vector_type = typename std::remove_reference<decltype(dest)>::type;
  //        using src_data_type = typename std::remove_pointer<decltype(src)>::type;
  //        using type = typename vector_type::value_type;
  //        static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
  //        cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  //    };

  outputGPU.allocate(num_rechits, ctx.stream());

  if (initCuda) {
    // Initialize persistent arrays for rechit positions
    persistentDataCPU.allocate(nValidDetIds, ctx.stream());
    persistentDataGPU.allocate(nValidDetIds, ctx.stream());
    scratchDataGPU.allocate(nValidDetIds, ctx.stream());

    uint32_t nRHTotal = 0;
    for (const auto& denseId : *vDenseIdHcal) {
      DetId detId = topology_.get()->denseId2detId(denseId);
      HcalDetId hid(detId.rawId());

      persistentDataCPU.rh_pos[nRHTotal] = make_float3(validDetIdPositions.at(nRHTotal).x(),
                                                       validDetIdPositions.at(nRHTotal).y(),
                                                       validDetIdPositions.at(nRHTotal).z());
      persistentDataCPU.rh_detId[nRHTotal] = hid.rawId();
      auto neigh = reinterpret_cast<PFRecHitHCALDenseIdNavigator*>(&(*navigator_))->getNeighbours(denseId);
      for (uint32_t n = 0; n < 8; n++) {
        // cmssdt.cern.ch/lxr/source/RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h#0087
        // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
        // neighboursHcal_[centerIndex][0] is the rechit itself. Skip for neighbour array
        // If no neighbour exists in a direction, the value will be 0
        // Some neighbors from HF included! Need to test if these are included in the map!
        //auto neighDetId = neighboursHcal_[centerIndex][n+1].rawId();
        auto neighDetId = neigh[n + 1].rawId();
        if (neighDetId > 0 && detIdToIndex.find(neighDetId) != detIdToIndex.end()) {
          persistentDataCPU.rh_neighbours[nRHTotal * 8 + n] = detIdToIndex[neighDetId];
        } else
          persistentDataCPU.rh_neighbours[nRHTotal * 8 + n] = -1;
      }
      nRHTotal++;
    }

    // Copy to GPU
    cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_pos.get(),
                              persistentDataCPU.rh_pos.get(),
                              nRHTotal * sizeof(float3),
                              cudaMemcpyHostToDevice,
                              ctx.stream()));
    cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_detId.get(),
                              persistentDataCPU.rh_detId.get(),
                              nRHTotal * sizeof(uint32_t),
                              cudaMemcpyHostToDevice,
                              ctx.stream()));
    cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_neighbours.get(),
                              persistentDataCPU.rh_neighbours.get(),
                              8 * nRHTotal * sizeof(int),
                              cudaMemcpyHostToDevice,
                              ctx.stream()));

    // Initialize Cuda constants
    PFRecHit::HCAL::initializeCudaConstants(nValidBarrelIds, nValidEndcapIds, qTestThresh);

    initCuda = false;
  }

  if (num_rechits == 0) {
    return;
  }

  // Entry point for GPU calls
  PFRecHit::HCAL::entryPoint(HBHERecHitSoA, outputGPU, persistentDataGPU, scratchDataGPU, ctx.stream(), GPU_timers);

  if (cudaStreamQuery(ctx.stream()) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(ctx.stream()));

  // Copy back PFRecHit SoA data to CPU
  auto lambdaToTransferSize = [&ctx](auto& dest, auto* src, auto size) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, size * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  nPFRHTotal = outputGPU.PFRecHits.size + outputGPU.PFRecHits.sizeCleaned;
  tmpPFRecHits.resize(nPFRHTotal);
  tmpPFRecHits.size = outputGPU.PFRecHits.size;
  tmpPFRecHits.sizeCleaned = outputGPU.PFRecHits.sizeCleaned;

  lambdaToTransferSize(tmpPFRecHits.pfrh_depth, outputGPU.PFRecHits.pfrh_depth.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_layer, outputGPU.PFRecHits.pfrh_layer.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_detId, outputGPU.PFRecHits.pfrh_detId.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbours, outputGPU.PFRecHits.pfrh_neighbours.get(), 8 * nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbourInfos, outputGPU.PFRecHits.pfrh_neighbourInfos.get(), 8 * nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_time, outputGPU.PFRecHits.pfrh_time.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_energy, outputGPU.PFRecHits.pfrh_energy.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_x, outputGPU.PFRecHits.pfrh_x.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_y, outputGPU.PFRecHits.pfrh_y.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_z, outputGPU.PFRecHits.pfrh_z.get(), nPFRHTotal);
  if (cudaStreamQuery(ctx.stream()) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(ctx.stream()));
}

void PFHBHERechitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(outputGPU.PFRecHits));
  auto pfrhLegacy = std::make_unique<reco::PFRecHitCollection>();
  auto pfrhLegacyCleaned = std::make_unique<reco::PFRecHitCollection>();

  const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  pfrhLegacy->reserve(tmpPFRecHits.size);
  pfrhLegacyCleaned->reserve(tmpPFRecHits.sizeCleaned);
  for (unsigned i = 0; i < nPFRHTotal; i++) {
    HcalDetId hid(tmpPFRecHits.pfrh_detId[i]);

    std::shared_ptr<const CaloCellGeometry> thisCell = nullptr;
    PFLayer::Layer layer = PFLayer::HCAL_BARREL1;
    switch (hid.subdet()) {
      case HcalBarrel:
        thisCell = hcalBarrelGeo->getGeometry(hid);
        layer = PFLayer::HCAL_BARREL1;
        break;

      case HcalEndcap:
        thisCell = hcalEndcapGeo->getGeometry(hid);
        layer = PFLayer::HCAL_ENDCAP;
        break;
      default:
        break;
    }
    reco::PFRecHit pfrh(thisCell, hid.rawId(), layer, tmpPFRecHits.pfrh_energy[i]);
    pfrh.setTime(tmpPFRecHits.pfrh_time[i]);
    pfrh.setDepth(hid.depth());

    std::vector<int> etas = {0, 1, 0, -1, 1, 1, -1, -1};
    std::vector<int> phis = {1, 1, -1, -1, 0, -1, 0, 1};
    std::vector<int> gpuOrder = {0, 4, 1, 5, 2, 6, 3, 7};
    for (int n = 0; n < 8; n++) {
      int neighId = tmpPFRecHits.pfrh_neighbours[i * 8 + gpuOrder[n]];
      if (i < tmpPFRecHits.size && neighId > -1 && neighId < (int)tmpPFRecHits.size)
        pfrh.addNeighbour(etas[n], phis[n], 0, neighId);
    }

    if (i < tmpPFRecHits.size)
      pfrhLegacy->push_back(pfrh);
    else
      pfrhLegacyCleaned->push_back(pfrh);
  }
  //#ifdef PF_DEBUG_ENABLE
  //  printf("pfrhLegacy->size() = %d\tpfrhLegacyCleaned->size() = %d\n\n", (int)pfrhLegacy->size(), (int)pfrhLegacyCleaned->size());
  //#endif

  event.put(std::move(pfrhLegacy), "");
  event.put(std::move(pfrhLegacyCleaned), "Cleaned");

  tmpPFRecHits.resize(0);

#ifdef PF_DEBUG_ENABLE
  for (int i = 0; i < (int)GPU_timers.size(); i++)
    hTimers->Fill(i, GPU_timers[i]);
  tree->Fill();
  numEvents++;
#endif
}

DEFINE_FWK_MODULE(PFHBHERechitProducerGPU);
