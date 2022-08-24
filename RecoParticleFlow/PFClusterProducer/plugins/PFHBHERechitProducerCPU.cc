#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "DeclsForKernels.h"
#include "SimplePFGPUAlgos.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "SimplePFGPUAlgos.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

// Comment out to disable debugging
#define PF_DEBUG_ENABLE

#ifdef PF_DEBUG_ENABLE
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#endif

#include <iostream>
#include <functional>
#include <optional>
#include <vector>
#include <array>



class PFHBHERechitProducerCPU : public edm::stream::EDProducer <edm::ExternalWork> {
public:
  explicit PFHBHERechitProducerCPU(edm::ParameterSet const&);
  ~PFHBHERechitProducerCPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  
private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

  // Input Product Type
  using PFRecHitSoAProductType = cms::cuda::Product<PFRecHit::HCAL::OutputPFRecHitDataGPU>;
  
  // Output Product Type
  
  const bool produceSoA_;           // PFRecHits in SoA format
  const bool produceLegacy_;        // PFRecHits in legacy format
  const bool produceCleanedLegacy_; // Cleaned PFRecHits in legacy format

  using IProductType = cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>;
  // Input Token
  const edm::EDGetTokenT<IProductType> InputPFRecHitSoA_Token_; 

  using OProductType = hcal::PFRecHitCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>>; 
  // Output Tokens
  edm::EDPutTokenT<OProductType> OutputPFRecHitSoA_Token_;
  edm::EDPutTokenT<reco::PFRecHitCollection> OutputPFRecHitLegacy_Token_;
  edm::EDPutTokenT<reco::PFRecHitCollection> OutputPFRecHitCleanedLegacy_Token_;

  cms::cuda::ContextState cudaState_;

  hcal::PFRecHitCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFRecHits;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESHandle<CaloGeometry> geoHandle;

  //uint32_t nPFRHTotal = 0;
  uint32_t nRechitsTotal = 0;
};


PFHBHERechitProducerCPU::PFHBHERechitProducerCPU(edm::ParameterSet const& ps)
    : produceSoA_{ps.getParameter<bool>("produceSoA")},
      produceLegacy_{ps.getParameter<bool>("produceLegacy")},
      produceCleanedLegacy_{ps.getParameter<bool>("produceCleanedLegacy")},
      InputPFRecHitSoA_Token_{consumes<IProductType>(ps.getParameterSetVector("producers")[0].getParameter<edm::InputTag>("src"))},
      OutputPFRecHitSoA_Token_{produceSoA_ ? produces<OProductType>(ps.getParameter<std::string>("PFRecHitsSoALabelOut"))
                                           : edm::EDPutTokenT<OProductType>{}}, // empty token if disabled
      OutputPFRecHitLegacy_Token_{produceLegacy_ 
                                    ? produces<reco::PFRecHitCollection>(ps.getParameter<std::string>("PFRecHitsLegacyLabelOut"))
                                    : edm::EDPutTokenT<reco::PFRecHitCollection>{}}, // empty token if disabled
      OutputPFRecHitCleanedLegacy_Token_{produceCleanedLegacy_ 
                                    ? produces<reco::PFRecHitCollection>(ps.getParameter<std::string>("PFRecHitsCleanedLegacyLabelOut"))
                                    : edm::EDPutTokenT<reco::PFRecHitCollection>{}}, // empty token if disabled
      geomToken_(esConsumes()) {}


PFHBHERechitProducerCPU::~PFHBHERechitProducerCPU() {
}

void PFHBHERechitProducerCPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("PFRecHitsGPULabelIn", edm::InputTag{"PFHBHERechitProducerGPU"});
    desc.add<std::string>("PFRecHitsSoALabelOut", "PFRecHitSoA");
    desc.add<std::string>("PFRecHitsLegacyLabelOut", "");
    desc.add<std::string>("PFRecHitsCleanedLegacyLabelOut", "Cleaned");
    desc.add<bool>("produceSoA", true);
    desc.add<bool>("produceLegacy", true);
    desc.add<bool>("produceCleanedLegacy", true);
 
//    cdesc.addWithDefaultLabel(desc);
    // Prevents the producer and navigator parameter sets from throwing an exception
    // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
    desc.setAllowAnything();

    cdesc.addWithDefaultLabel(desc);
}

void PFHBHERechitProducerCPU::acquire(edm::Event const& event,
				      edm::EventSetup const& setup,
				      edm::WaitingTaskWithArenaHolder holder) {
    
    //auto start = std::chrono::high_resolution_clock::now();

    auto const& PFHBHERecHitSoAProduct = event.get(InputPFRecHitSoA_Token_);
    cms::cuda::ScopedContextAcquire ctx{PFHBHERecHitSoAProduct, std::move(holder), cudaState_};
    auto const& PFHBHERecHitSoA = ctx.get(PFHBHERecHitSoAProduct);
//    size_t num_rechits = PFHBHERecHitSoA.size;
//    tmpPFRecHits.resize(num_rechits);
    
    auto lambdaToTransferSize = [&ctx](auto& dest, auto* src, auto size) {
        using vector_type = typename std::remove_reference<decltype(dest)>::type;
        using src_data_type = typename std::remove_pointer<decltype(src)>::type;
        using type = typename vector_type::value_type;
        static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
        cudaCheck(cudaMemcpyAsync(dest.data(), src, size * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
    };  

  uint32_t offset = 0;  // Offset for first PFRecHit to copy
  nRechitsTotal = 0;   // Total number of PFRecHits to copy

  if (produceSoA_)
    nRechitsTotal = PFHBHERecHitSoA.size + PFHBHERecHitSoA.sizeCleaned;
  else if (produceLegacy_ && produceCleanedLegacy_)
    nRechitsTotal = PFHBHERecHitSoA.size + PFHBHERecHitSoA.sizeCleaned;
  else if (produceLegacy_ && PFHBHERecHitSoA.size > 0) {
    // Passing PFRecHits only
    nRechitsTotal = PFHBHERecHitSoA.size;
  }
  else if (produceCleanedLegacy_ && PFHBHERecHitSoA.sizeCleaned > 0) {
    // Cleaned PFRecHits only
    offset = PFHBHERecHitSoA.size; 
    nRechitsTotal = PFHBHERecHitSoA.sizeCleaned;
  }
  else {
    // Nothing to do. Exit
    return;
  }

  if (nRechitsTotal == 0)
    return;

  tmpPFRecHits.resize(nRechitsTotal);
  tmpPFRecHits.size = PFHBHERecHitSoA.size;                 // Total number of rechits passing cuts: size
  tmpPFRecHits.sizeCleaned = PFHBHERecHitSoA.sizeCleaned;   // Total number of rechits cleaned (failing cuts): sizeCleaned
  
  lambdaToTransferSize(tmpPFRecHits.pfrh_depth, PFHBHERecHitSoA.pfrh_depth.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_layer, PFHBHERecHitSoA.pfrh_layer.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_detId, PFHBHERecHitSoA.pfrh_detId.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbours, PFHBHERecHitSoA.pfrh_neighbours.get() + 8*offset, 8*nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbourInfos, PFHBHERecHitSoA.pfrh_neighbourInfos.get() + 8*offset, 8*nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_time, PFHBHERecHitSoA.pfrh_time.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_energy, PFHBHERecHitSoA.pfrh_energy.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_x, PFHBHERecHitSoA.pfrh_x.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_y, PFHBHERecHitSoA.pfrh_y.get() + offset, nRechitsTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_z, PFHBHERecHitSoA.pfrh_z.get() + offset, nRechitsTotal);
//  if (cudaStreamQuery(ctx.stream()) != cudaSuccess) cudaCheck(cudaStreamSynchronize(ctx.stream()));
}


void PFHBHERechitProducerCPU::produce(edm::Event& event, edm::EventSetup const& setup) {

  geoHandle = setup.getHandle(geomToken_);
  const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);
  if (produceLegacy_ || produceCleanedLegacy_) {
      auto pfrhLegacy = std::make_unique<reco::PFRecHitCollection>();
      auto pfrhCleanedLegacy = std::make_unique<reco::PFRecHitCollection>();
      
      pfrhLegacy->reserve(tmpPFRecHits.size);
      pfrhCleanedLegacy->reserve(tmpPFRecHits.sizeCleaned);
      for (unsigned i = 0; i < nRechitsTotal; i++) {
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
        
        std::vector<int> etas     = {0,  1,  0, -1,  1,  1, -1, -1};
        std::vector<int> phis     = {1,  1, -1, -1,  0, -1,  0,  1};
        std::vector<int> gpuOrder = {0,  4,  1,  5,  2,  6,  3,  7};
        for (int n = 0; n < 8; n++) {
            int neighId = tmpPFRecHits.pfrh_neighbours[i*8+gpuOrder[n]];
            if (i < tmpPFRecHits.size && neighId > -1 && neighId < (int)tmpPFRecHits.size)
                pfrh.addNeighbour(etas[n], phis[n], 0, neighId); 
        }

        if (i < tmpPFRecHits.size)
            pfrhLegacy->push_back(pfrh);
        else
            pfrhCleanedLegacy->push_back(pfrh);
      }

    #ifdef PF_DEBUG_ENABLE
      printf("pfrhLegacy->size() = %d\tpfrhCleanedLegacy->size() = %d\n\n", (int)pfrhLegacy->size(), (int)pfrhCleanedLegacy->size());
    #endif
    if (produceLegacy_)
        event.put(OutputPFRecHitLegacy_Token_, std::move(pfrhLegacy));
    if (produceCleanedLegacy_)
        event.put(OutputPFRecHitCleanedLegacy_Token_, std::move(pfrhCleanedLegacy));
//      event.put(std::move(pfrhLegacy), "");
//      event.put(std::move(pfrhLegacyCleaned), "Cleaned");
  }
  if (produceSoA_)
    event.emplace(OutputPFRecHitSoA_Token_, std::move(tmpPFRecHits));

  tmpPFRecHits.resize(0);
}

DEFINE_FWK_MODULE(PFHBHERechitProducerCPU);
