#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

class SiPixelRecHitFromSOA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitFromSOA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitFromSOA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HMSstorage = HostProduct<unsigned int[]>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> tokenHit_;  // CUDA hits
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;           // Legacy Clusters

  uint32_t m_nHits;
  cms::cuda::host::unique_ptr<uint16_t[]> m_store16;
  cms::cuda::host::unique_ptr<float[]> m_store32;
  cms::cuda::host::unique_ptr<uint32_t[]> m_hitsModuleStart;
};

SiPixelRecHitFromSOA::SiPixelRecHitFromSOA(const edm::ParameterSet& iConfig)
    : tokenHit_(
          consumes<cms::cuda::Product<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      clusterToken_(consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<SiPixelRecHitCollectionNew>();
  produces<HMSstorage>();
}

void SiPixelRecHitFromSOA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  descriptions.add("siPixelRecHitFromSOA", desc);
}

void SiPixelRecHitFromSOA::acquire(edm::Event const& iEvent,
                                   edm::EventSetup const& iSetup,
                                   edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<TrackingRecHit2DCUDA> const& inputDataWrapped = iEvent.get(tokenHit_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_nHits = inputData.nHits();

  // std::cout<< "converting " << m_nHits << " Hits"<< std::endl;

  if (0 == m_nHits)
    return;
  m_store32 = inputData.localCoordToHostAsync(ctx.stream());
  //  m_store16 = inputData.detIndexToHostAsync(ctx.stream();
  m_hitsModuleStart = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitFromSOA::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  // yes a unique ptr of a unique ptr so edm is happy
  auto sizeOfHitModuleStart = gpuClustering::MaxNumModules + 1;
  auto hmsp = std::make_unique<uint32_t[]>(sizeOfHitModuleStart);
  std::copy(m_hitsModuleStart.get(), m_hitsModuleStart.get() + sizeOfHitModuleStart, hmsp.get());
  auto hms = std::make_unique<HMSstorage>(std::move(hmsp));  // hmsp is gone
  iEvent.put(std::move(hms));                                // hms is gone!

  auto output = std::make_unique<SiPixelRecHitCollectionNew>();
  if (0 == m_nHits) {
    iEvent.put(std::move(output));
    return;
  }

  auto xl = m_store32.get();
  auto yl = xl + m_nHits;
  auto xe = yl + m_nHits;
  auto ye = xe + m_nHits;

  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom);
  geom = geom.product();

  edm::Handle<SiPixelClusterCollectionNew> hclusters;
  iEvent.getByToken(clusterToken_, hclusters);

  auto const& input = *hclusters;

  constexpr uint32_t MaxHitsInModule = gpuClustering::MaxHitsInModule;

  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
    numberOfDetUnits++;
    unsigned int detid = DSViter->detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(*output, detid);
    auto fc = m_hitsModuleStart[gind];
    auto lc = m_hitsModuleStart[gind + 1];
    auto nhits = lc - fc;

    assert(lc > fc);
    // std::cout << "in det " << gind << ": conv " << nhits << " hits from " << DSViter->size() << " legacy clusters"
    //          <<' '<< fc <<','<<lc<<std::endl;
    if (nhits > MaxHitsInModule)
      printf(
          "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nhits, gind, MaxHitsInModule);
    nhits = std::min(nhits, MaxHitsInModule);

    //std::cout << "in det " << gind << "conv " << nhits << " hits from " << DSViter->size() << " legacy clusters"
    //          <<' '<< lc <<','<<fc<<std::endl;

    if (0 == nhits)
      continue;
    auto jnd = [&](int k) { return fc + k; };
    assert(nhits <= DSViter->size());
    if (nhits != DSViter->size()) {
      edm::LogWarning("GPUHits2CPU") << "nhits!= nclus " << nhits << ' ' << DSViter->size() << std::endl;
    }
    for (auto const& clust : *DSViter) {
      assert(clust.originalId() >= 0);
      assert(clust.originalId() < DSViter->size());
      if (clust.originalId() >= nhits)
        continue;
      auto ij = jnd(clust.originalId());
      if (ij >= TrackingRecHit2DSOAView::maxHits())
        continue;  // overflow...
      LocalPoint lp(xl[ij], yl[ij]);
      LocalError le(xe[ij], 0, ye[ij]);
      SiPixelRecHitQuality::QualWordType rqw = 0;

      numberOfClusters++;

      /*   cpu version....  (for reference)
           std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
           LocalPoint lp( std::get<0>(tuple) );
           LocalError le( std::get<1>(tuple) );
           SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      */

      // Create a persistent edm::Ref to the cluster
      edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> cluster = edmNew::makeRefTo(hclusters, &clust);
      // Make a RecHit and add it to the DetSet
      SiPixelRecHit hit(lp, le, rqw, *genericDet, cluster);
      //
      // Now save it =================
      recHitsOnDetUnit.push_back(hit);
      // =============================

      // std::cout << "SiPixelRecHitGPUVI " << numberOfClusters << ' '<< lp << " " << le << std::endl;

    }  //  <-- End loop on Clusters

    //  LogDebug("SiPixelRecHitGPU")
    //std::cout << "SiPixelRecHitGPUVI "
    //	<< " Found " << recHitsOnDetUnit.size() << " RecHits on " << detid //;
    // << std::endl;

  }  //    <-- End loop on DetUnits

  /*
  std::cout << "SiPixelRecHitGPUVI $ det, clus, lost "
    <<  numberOfDetUnits << ' '
    << numberOfClusters  << ' '
    << std::endl;
  */

  iEvent.put(std::move(output));
}

DEFINE_FWK_MODULE(SiPixelRecHitFromSOA);
