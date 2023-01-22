#include <array>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

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
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitParamsGPU.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHETopologyGPU.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHETopologyGPURcd.h"

#include "DeclsForKernels.h"
#include "SimplePFGPUAlgos.h"

typedef PFHCALDenseIdNavigator<HcalDetId, HcalTopology, false> PFRecHitHCALDenseIdNavigator;

class PFHBHERecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PFHBHERecHitProducerGPU(edm::ParameterSet const&);
  ~PFHBHERecHitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  const bool produceSoA_;            // PFRecHits in SoA format
  const bool produceLegacy_;         // PFRecHits in legacy format
  const bool produceCleanedLegacy_;  // Cleaned PFRecHits in legacy format
  const bool fullLegacy_ = false; // Store full information to legacy format data

  //Output Product Type
  using PFRecHitSoAProductType = cms::cuda::Product<PFRecHit::HCAL::OutputPFRecHitDataGPU>;
  //Input Token
  using IProductType = cms::cuda::Product<hcal::RecHitCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<IProductType> InputRecHitSoA_Token_;
  //Output Token
  using OProductType = cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>;
  edm::EDPutTokenT<OProductType> OutputPFRecHitSoA_Token_;

  PFRecHit::HCAL::OutputPFRecHitDataGPU outputGPU;
  cms::cuda::ContextState cudaState_;

  // PFRecHits for GPU
  hcal::PFRecHitCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFRecHits;

  std::unique_ptr<PFRecHitNavigatorBase> navigator_;

  // HCAL geometry/topology
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESHandle<CaloGeometry> geoHandle;
  edm::ESHandle<HcalTopology> topoHandle;

  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_;
  std::unique_ptr<const HcalTopology> topology_;

  // PFHBHERecHit params
  const edm::ESGetToken<PFHBHERecHitParamsGPU, JobConfigurationGPURecord> recoParamsToken_;
  edm::ESHandle<PFHBHERecHitParamsGPU> recHitParametersHandle_;

  const edm::ESGetToken<PFHBHETopologyGPU, PFHBHETopologyGPURcd> topologyToken_;
  edm::ESHandle<PFHBHETopologyGPU> topologyHandle_;

  // Miscellaneous
  PFRecHit::HCAL::PersistentDataCPU persistentDataCPU;
  PFRecHit::HCAL::ScratchDataGPU scratchDataGPU;
  PFRecHit::HCAL::Constants cudaConstants;
  uint32_t nValidDetIds = 0;
  uint32_t nDenseIdsInRange = 0;
  std::vector<std::vector<DetId>>* neighboursHcal_;
  std::vector<unsigned>* vDenseIdHcal;
  std::vector<GlobalPoint> validDetIdPositions;
  unsigned denseIdHcalMax_ = 0;
  unsigned denseIdHcalMin_ = 0;
  std::unordered_map<unsigned, std::shared_ptr<const CaloCellGeometry>>
      detIdToCell;  // Mapping of detId to cell geometry.

  std::array<float, 5> GPU_timers;

  bool debug=false;

  unsigned int getIdx(const unsigned int denseid) const {
    unsigned index = denseid - denseIdHcalMin_;
    return index;
  }

};

PFHBHERecHitProducerGPU::PFHBHERecHitProducerGPU(edm::ParameterSet const& ps)
    : produceSoA_{ps.getParameter<bool>("produceSoA")},
      produceLegacy_{ps.getParameter<bool>("produceLegacy")},
      produceCleanedLegacy_{ps.getParameter<bool>("produceCleanedLegacy")},
      InputRecHitSoA_Token_{consumes<IProductType>(
          ps.getParameterSetVector("producers")[0].getParameter<edm::InputTag>("src"))},
      OutputPFRecHitSoA_Token_{produces<OProductType>(ps.getParameter<std::string>("PFRecHitsGPUOut"))},
      hcalToken_(esConsumes<edm::Transition::BeginRun>()),
      geomToken_(esConsumes<edm::Transition::BeginRun>()),
      recoParamsToken_{esConsumes()},
      topologyToken_{esConsumes()}{
  edm::ConsumesCollector cc = consumesCollector();

  produces<reco::PFRecHitCollection>();
  produces<reco::PFRecHitCollection>("Cleaned");

  // producer-related parameters
  const auto& prodConf = ps.getParameterSetVector("producers")[0];
  const std::string& prodName = prodConf.getParameter<std::string>("name");
  const auto& qualityConf = prodConf.getParameterSetVector("qualityTests")[0];

  // Single threshold
  const std::string& qualityTestName = qualityConf.getParameter<std::string>("name");

  // Thresholds vs depth
  const auto& qualityCutConfs = qualityConf.getParameterSetVector("cuts");
  //
  // navigator-related parameters
  const auto& navSet = ps.getParameterSet("navigator");
  navigator_ = PFRecHitNavigationFactory::get()->create(navSet.getParameter<std::string>("name"), navSet, cc);
}

PFHBHERecHitProducerGPU::~PFHBHERecHitProducerGPU() {
  topology_.release();
}

void PFHBHERecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("produceSoA", true);
  desc.add<bool>("produceLegacy", true);
  desc.add<bool>("produceCleanedLegacy", true);

  desc.add<std::string>("PFRecHitsGPUOut", "");
  // Prevents the producer and navigator parameter sets from throwing an exception
  // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
  desc.setAllowAnything();

  cdesc.addWithDefaultLabel(desc);
}

void PFHBHERecHitProducerGPU::beginRun(edm::Run const& r, edm::EventSetup const& setup) {
  navigator_->init(setup);
  if (!theRecNumberWatcher_.check(setup))
    return;

  topoHandle = setup.getHandle(hcalToken_);
  topology_.release();
  topology_.reset(topoHandle.product());

  //
  // Get list of valid Det Ids for HCAL barrel & endcap once
  geoHandle = setup.getHandle(geomToken_);
  // get the hcal geometry
  const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

  const std::vector<DetId>& validBarrelDetIds = hcalBarrelGeo->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& validEndcapDetIds = hcalEndcapGeo->getValidDetIds(DetId::Hcal, HcalEndcap);

  cudaConstants.nValidBarrelIds = validBarrelDetIds.size();
  cudaConstants.nValidEndcapIds = validEndcapDetIds.size();
  nValidDetIds = cudaConstants.nValidBarrelIds + cudaConstants.nValidEndcapIds;
  cudaConstants.nValidDetIds = nValidDetIds;

  vDenseIdHcal = reinterpret_cast<PFRecHitHCALDenseIdNavigator*>(&(*navigator_))->getValidDenseIds();
  //std::cout << "Found vDenseIdHcal->size() = " << vDenseIdHcal->size() << std::endl;

  // Fill a vector of cell neighbours
  denseIdHcalMax_ = *std::max_element(vDenseIdHcal->begin(), vDenseIdHcal->end());
  denseIdHcalMin_ = *std::min_element(vDenseIdHcal->begin(), vDenseIdHcal->end());
  //std::cout << denseIdHcalMax_ << " " << denseIdHcalMin_ << std::endl;
  nDenseIdsInRange = denseIdHcalMax_ - denseIdHcalMin_ + 1;
  cudaConstants.nDenseIdsInRange = nDenseIdsInRange;
  cudaConstants.denseIdHcalMin = denseIdHcalMin_;


  validDetIdPositions.clear();
  validDetIdPositions.reserve(nValidDetIds);
  detIdToCell.clear();
  detIdToCell.reserve(nValidDetIds);

  for (const auto& denseid : *vDenseIdHcal) {
    DetId detid_c = topology_.get()->denseId2detId(denseid);
    HcalDetId hid_c = HcalDetId(detid_c);

    //DetId detId = topology_.get()->denseId2detId(denseId);
    //HcalDetId hid(detId.rawId());

    if (hid_c.subdet() == HcalBarrel)
      validDetIdPositions.emplace_back(hcalBarrelGeo->getGeometry(detid_c)->getPosition());
    else if (hid_c.subdet() == HcalEndcap)
      validDetIdPositions.emplace_back(hcalEndcapGeo->getGeometry(detid_c)->getPosition());
    else
      std::cout << "Invalid subdetector found for detId " << hid_c.rawId() << ": " << hid_c.subdet() << std::endl;

    std::shared_ptr<const CaloCellGeometry> thisCell = nullptr;
    //PFLayer::Layer layer = PFLayer::HCAL_BARREL1;
    switch (hid_c.subdet()) {
    case HcalBarrel:
      thisCell = hcalBarrelGeo->getGeometry(hid_c);
      //layer = PFLayer::HCAL_BARREL1;
      break;

    case HcalEndcap:
      thisCell = hcalEndcapGeo->getGeometry(hid_c);
      //layer = PFLayer::HCAL_ENDCAP;
      break;
    default:
      break;
    }

    detIdToCell[hid_c.rawId()] = thisCell;

  }
  // -> vDenseIdHcal, validDetIdPositions

  //initCuda = true;  // (Re)initialize cuda arrays
  //KenH: for now comment this out, as we know we don't change the channel status on lumisection basis
}

void PFHBHERecHitProducerGPU::acquire(edm::Event const& event,
                                      edm::EventSetup const& setup,
                                      edm::WaitingTaskWithArenaHolder holder) {

  if (debug) std::cout << "PFHBHERecHitProducerGPU::acquire" << std::endl;

  auto const& HBHERecHitSoAProduct = event.get(InputRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{HBHERecHitSoAProduct, std::move(holder), cudaState_};
  auto const& HBHERecHitSoA = ctx.get(HBHERecHitSoAProduct);
  size_t num_rechits = HBHERecHitSoA.size;

  //
  //auto const& pulseOffsets = setup.getData(recoParamsToken_);
  //auto const& pulseOffsetsProduct = pulseOffsets.getProduct(ctx.stream());
  recHitParametersHandle_ = setup.getHandle(recoParamsToken_);
  //auto const& recHitParametersProduct2 = recHitParametersHandle_->getProduct(ctx.stream()); // to be passed to CUDA

  auto const& recHitParams = setup.getData(recoParamsToken_);
  auto const& recHitParamsProduct = recHitParams.getProduct(ctx.stream());

  topologyHandle_ = setup.getHandle(topologyToken_);
  auto const& topoData = setup.getData(topologyToken_);
  auto const& topoDataProduct = topoData.getProduct(ctx.stream());

  if (debug) {
  std::cout << (topoData.getValuesDetId()).size() << std::endl;

  std::cout << (recHitParametersHandle_->getValuesDepthHB())[0] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHB())[1] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHB())[2] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHB())[3] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHB()).size() << std::endl;

  std::cout << (recHitParams.getValuesDepthHB()).size() << std::endl;

  std::cout << (recHitParametersHandle_->getValuesDepthHE())[0] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE())[1] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE())[2] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE())[3] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE())[4] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE())[5] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE())[6] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesDepthHE()).size() << std::endl;

  std::cout << (recHitParametersHandle_->getValuesThresholdE_HB())[0] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HB())[1] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HB())[2] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HB())[3] << std::endl;

  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[0] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[1] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[2] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[3] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[4] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[5] << std::endl;
  std::cout << (recHitParametersHandle_->getValuesThresholdE_HE())[6] << std::endl;

  std::cout << "init starts" << std::endl;

  }

  scratchDataGPU.allocate(nValidDetIds, ctx.stream()); //Initialize scratchData array

  if (debug) std::cout << "init done" << std::endl;

  if (num_rechits == 0) return; // if no rechit, there is nothing to do.

  outputGPU.allocate(num_rechits, ctx.stream());

  // bundle up constants
  PFRecHit::HCAL::ConstantProducts constantProducts{
        recHitParamsProduct,
	recHitParams.getValuesDepthHB(),
	recHitParams.getValuesDepthHE(),
	recHitParams.getValuesThresholdE_HB(),
	recHitParams.getValuesThresholdE_HE(),
	topoDataProduct,
	topoData.getValuesDenseId(),
	topoData.getValuesDetId(),
	topoData.getValuesPosition(),
	topoData.getValuesNeighbours()
  };

  // Entry point for GPU calls
  GPU_timers.fill(0.0);
  PFRecHit::HCAL::entryPoint(HBHERecHitSoA, 
			                constantProducts,
			                outputGPU, 
                            scratchDataGPU, ctx.stream(), GPU_timers);

  // if (cudaStreamQuery(ctx.stream()) != cudaSuccess)
  //   cudaCheck(cudaStreamSynchronize(ctx.stream()));

  if (!produceLegacy_ && !produceCleanedLegacy_) return; // do device->host transfer only when we are producing Legacy data

  // Copy back PFRecHit SoA data to CPU
  auto lambdaToTransferSize = [&ctx](auto& dest, auto* src, auto size) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, size * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  num_rechits = outputGPU.PFRecHits.size + outputGPU.PFRecHits.sizeCleaned; // transfer only what become PFRecHits
  tmpPFRecHits.resize(num_rechits);
  lambdaToTransferSize(tmpPFRecHits.pfrh_detId, outputGPU.PFRecHits.pfrh_detId.get(), num_rechits);
  if (fullLegacy_)
    lambdaToTransferSize(tmpPFRecHits.pfrh_neighbours, outputGPU.PFRecHits.pfrh_neighbours.get(), 8 * num_rechits);
  lambdaToTransferSize(tmpPFRecHits.pfrh_time, outputGPU.PFRecHits.pfrh_time.get(), num_rechits);
  lambdaToTransferSize(tmpPFRecHits.pfrh_energy, outputGPU.PFRecHits.pfrh_energy.get(), num_rechits);
  // if (cudaStreamQuery(ctx.stream()) != cudaSuccess)
  //   cudaCheck(cudaStreamSynchronize(ctx.stream()));

}

void PFHBHERecHitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {

  cms::cuda::ScopedContextProduce ctx{cudaState_};
  if (produceSoA_)
    ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(outputGPU.PFRecHits));

  if (produceLegacy_ || produceCleanedLegacy_){

    auto pfrhLegacy = std::make_unique<reco::PFRecHitCollection>();
    auto pfrhLegacyCleaned = std::make_unique<reco::PFRecHitCollection>();

    //Use pre-filled unordered_map, but we may go back to directly using geometry
    //const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
    //const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

    auto nPFRHTotal = outputGPU.PFRecHits.size + outputGPU.PFRecHits.sizeCleaned;
    tmpPFRecHits.size = outputGPU.PFRecHits.size;
    tmpPFRecHits.sizeCleaned = outputGPU.PFRecHits.sizeCleaned;

    pfrhLegacy->reserve(tmpPFRecHits.size);
    if (produceCleanedLegacy_)
      pfrhLegacyCleaned->reserve(tmpPFRecHits.sizeCleaned);

    for (unsigned i = 0; i < nPFRHTotal; i++) {
      HcalDetId hid(tmpPFRecHits.pfrh_detId[i]);

      //std::shared_ptr<const CaloCellGeometry> thisCell = nullptr;
      PFLayer::Layer layer = PFLayer::HCAL_BARREL1;
      switch (hid.subdet()) {
        case HcalBarrel:
	   //thisCell = hcalBarrelGeo->getGeometry(hid);
	  layer = PFLayer::HCAL_BARREL1;
	  break;

        case HcalEndcap:
	  layer = PFLayer::HCAL_ENDCAP;
	  //thisCell = hcalEndcapGeo->getGeometry(hid);
	  break;
        default:
	  break;
      }

      reco::PFRecHit pfrh(detIdToCell.find(hid.rawId())->second, hid.rawId(), layer, tmpPFRecHits.pfrh_energy[i]);
      pfrh.setTime(tmpPFRecHits.pfrh_time[i]);
      pfrh.setDepth(hid.depth());

      // store full PF rechits including neighbor info (neighbor info is not necessary in legacy format when PFCluster is produced on GPU)
      if (fullLegacy_) {
      std::vector<int> etas = {0, 1, 0, -1, 1, 1, -1, -1};
      std::vector<int> phis = {1, 1, -1, -1, 0, -1, 0, 1};
      std::vector<int> gpuOrder = {0, 4, 1, 5, 2, 6, 3, 7};
      for (int n = 0; n < 8; n++) {
	int neighId = tmpPFRecHits.pfrh_neighbours[i * 8 + gpuOrder[n]];
	if (i < tmpPFRecHits.size && neighId > -1 && neighId < (int)tmpPFRecHits.size)
	  pfrh.addNeighbour(etas[n], phis[n], 0, neighId);
      }
      } // fullLegacy

      if (i < tmpPFRecHits.size)
	pfrhLegacy->push_back(pfrh);
      else
	if (produceCleanedLegacy_)
	  pfrhLegacyCleaned->push_back(pfrh);
    }

    if (produceLegacy_) event.put(std::move(pfrhLegacy), "");
    if (produceCleanedLegacy_) event.put(std::move(pfrhLegacyCleaned), "Cleaned");

    //tmpPFRecHits.resize(0); // clear the temporary collection for the next event
    //KenH: comment out for now
  } // if (produceLegacy_ || produceCleanedLegacy_)
}

DEFINE_FWK_MODULE(PFHBHERecHitProducerGPU);
