#include <cuda_runtime.h>

#include <fmt/printf.h>

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

class SiPixelRecHitFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitFromCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HMSstorage = HostProduct<uint32_t[]>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> hitsToken_;  // CUDA hits
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;            // legacy clusters
  const edm::EDPutTokenT<SiPixelRecHitCollection> rechitsPutToken_;             // legacy rechits
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;

  uint32_t nHits_;
  cms::cuda::host::unique_ptr<float[]> store32_;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStart_;
};

SiPixelRecHitFromCUDA::SiPixelRecHitFromCUDA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      hitsToken_(
          consumes<cms::cuda::Product<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      clusterToken_(consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"))),
      rechitsPutToken_(produces<SiPixelRecHitCollection>()),
      hostPutToken_(produces<HMSstorage>()) {}

void SiPixelRecHitFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelRecHitFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<TrackingRecHit2DCUDA> const& inputDataWrapped = iEvent.get(hitsToken_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  nHits_ = inputData.nHits();

  LogDebug("SiPixelRecHitFromCUDA") << "converting " << nHits_ << " Hits";

  if (0 == nHits_)
    return;
  store32_ = inputData.localCoordToHostAsync(ctx.stream());
  hitsModuleStart_ = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  // allocate a buffer for the indices of the clusters
  auto hmsp = std::make_unique<uint32_t[]>(gpuClustering::maxNumModules + 1);
  std::copy(hitsModuleStart_.get(), hitsModuleStart_.get() + gpuClustering::maxNumModules + 1, hmsp.get());
  // wrap the buffer in a HostProduct, and move it to the Event, without reallocating the buffer or affecting hitsModuleStart
  iEvent.emplace(hostPutToken_, std::move(hmsp));

  SiPixelRecHitCollection output;
  if (0 == nHits_) {
    iEvent.emplace(rechitsPutToken_, std::move(output));
    return;
  }
  output.reserve(gpuClustering::maxNumModules, nHits_);

  auto xl = store32_.get();
  auto yl = xl + nHits_;
  auto xe = yl + nHits_;
  auto ye = xe + nHits_;

  const TrackerGeometry* geom = &es.getData(geomToken_);

  edm::Handle<SiPixelClusterCollectionNew> hclusters = iEvent.getHandle(clusterToken_);
  auto const& input = *hclusters;

  constexpr uint32_t maxHitsInModule = gpuClustering::maxHitsInModule();

  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  for (auto const& dsv : input) {
    numberOfDetUnits++;
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    SiPixelRecHitCollection::FastFiller recHitsOnDetUnit(output, detid);
    auto fc = hitsModuleStart_[gind];
    auto lc = hitsModuleStart_[gind + 1];
    auto nhits = lc - fc;

    assert(lc > fc);
    LogDebug("SiPixelRecHitFromCUDA") << "in det " << gind << ": conv " << nhits << " hits from " << dsv.size()
                                      << " legacy clusters" << ' ' << fc << ',' << lc;
    if (nhits > maxHitsInModule)
      edm::LogWarning("SiPixelRecHitFromCUDA") << fmt::sprintf(
          "Too many clusters %d in module %d. Only the first %d hits will be converted", nhits, gind, maxHitsInModule);
    nhits = std::min(nhits, maxHitsInModule);

    LogDebug("SiPixelRecHitFromCUDA") << "in det " << gind << "conv " << nhits << " hits from " << dsv.size()
                                      << " legacy clusters" << ' ' << lc << ',' << fc;

    if (0 == nhits)
      continue;
    auto jnd = [&](int k) { return fc + k; };
    assert(nhits <= dsv.size());
    if (nhits != dsv.size()) {
      edm::LogWarning("GPUHits2CPU") << "nhits!= nclus " << nhits << ' ' << dsv.size();
    }
    for (auto const& clust : dsv) {
      assert(clust.originalId() >= 0);
      assert(clust.originalId() < dsv.size());
      if (clust.originalId() >= nhits)
        continue;
      auto ij = jnd(clust.originalId());
      LocalPoint lp(xl[ij], yl[ij]);
      LocalError le(xe[ij], 0, ye[ij]);
      SiPixelRecHitQuality::QualWordType rqw = 0;

      numberOfClusters++;

      /* cpu version....  (for reference)
      std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
      LocalPoint lp( std::get<0>(tuple) );
      LocalError le( std::get<1>(tuple) );
      SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      */

      // Create a persistent edm::Ref to the cluster
      edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> cluster = edmNew::makeRefTo(hclusters, &clust);
      // Make a RecHit and add it to the DetSet
      recHitsOnDetUnit.emplace_back(lp, le, rqw, *genericDet, cluster);
      // =============================

      LogDebug("SiPixelRecHitFromCUDA") << "cluster " << numberOfClusters << " at " << lp << ' ' << le;

    }  //  <-- End loop on Clusters

    //  LogDebug("SiPixelRecHitGPU")
    LogDebug("SiPixelRecHitFromCUDA") << "found " << recHitsOnDetUnit.size() << " RecHits on " << detid;

  }  //    <-- End loop on DetUnits

  LogDebug("SiPixelRecHitFromCUDA") << "found " << numberOfDetUnits << " dets, " << numberOfClusters << " clusters";

  iEvent.emplace(rechitsPutToken_, std::move(output));
}

DEFINE_FWK_MODULE(SiPixelRecHitFromCUDA);
