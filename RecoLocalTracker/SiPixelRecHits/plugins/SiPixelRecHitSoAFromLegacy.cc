#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "gpuPixelRecHits.h"

template <typename TrackerTraits>
class SiPixelRecHitSoAFromLegacyT : public edm::global::EDProducer<> {
public:
  explicit SiPixelRecHitSoAFromLegacyT(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAFromLegacyT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HitModuleStart = std::array<uint32_t, TrackerTraits::numberOfModules + 1>;
  using HMSstorage = HostProduct<uint32_t[]>;
  using HitsOnHost = TrackingRecHitSoAHost<TrackerTraits>;

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;  // Legacy Clusters
  const edm::EDPutTokenT<HitsOnHost> tokenHit_;
  const edm::EDPutTokenT<HMSstorage> tokenModuleStart_;
  const bool convert2Legacy_;
};

template <typename TrackerTraits>
SiPixelRecHitSoAFromLegacyT<TrackerTraits>::SiPixelRecHitSoAFromLegacyT(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
      bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      clusterToken_{consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"))},
      tokenHit_{produces<HitsOnHost>()},
      tokenModuleStart_{produces<HMSstorage>()},
      convert2Legacy_(iConfig.getParameter<bool>("convertToLegacy")) {
  if (convert2Legacy_)
    produces<SiPixelRecHitCollectionNew>();
}

template <typename TrackerTraits>
void SiPixelRecHitSoAFromLegacyT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  std::string cpeName = "PixelCPEFast";
  cpeName += TrackerTraits::nameModifier;
  desc.add<std::string>("CPE", cpeName);
  desc.add<bool>("convertToLegacy", false);

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiPixelRecHitSoAFromLegacyT<TrackerTraits>::produce(edm::StreamID streamID,
                                                         edm::Event& iEvent,
                                                         const edm::EventSetup& es) const {
  const TrackerGeometry* geom_ = &es.getData(geomToken_);
  PixelCPEFast<TrackerTraits> const* fcpe = dynamic_cast<const PixelCPEFast<TrackerTraits>*>(&es.getData(cpeToken_));
  if (not fcpe) {
    throw cms::Exception("Configuration") << "SiPixelRecHitSoAFromLegacy can only use a CPE of type PixelCPEFast";
  }
  auto const& cpeView = fcpe->getCPUProduct();

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

  BeamSpotPOD bsHost;
  bsHost.x = bs.x0();
  bsHost.y = bs.y0();
  bsHost.z = bs.z0();

  edm::Handle<SiPixelClusterCollectionNew> hclusters;
  iEvent.getByToken(clusterToken_, hclusters);
  auto const& input = *hclusters;

  constexpr int nModules = TrackerTraits::numberOfModules;
  constexpr int startBPIX2 = pixelTopology::layerStart<TrackerTraits>(1);

  // allocate a buffer for the indices of the clusters
  auto hmsp = std::make_unique<uint32_t[]>(nModules + 1);
  auto hitsModuleStart = hmsp.get();
  // wrap the buffer in a HostProduct
  auto hms = std::make_unique<HMSstorage>(std::move(hmsp));
  // move the HostProduct to the Event, without reallocating the buffer or affecting hitsModuleStart
  iEvent.put(tokenModuleStart_, std::move(hms));

  // legacy output
  auto legacyOutput = std::make_unique<SiPixelRecHitCollectionNew>();

  std::vector<edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>> clusterRef;

  constexpr uint32_t maxHitsInModule = TrackerTraits::maxNumClustersPerModules;

  cms::cuda::PortableHostCollection<SiPixelClustersCUDALayout<>> clusters_h(nModules + 1);

  memset(clusters_h.view().clusInModule(), 0, (nModules + 1) * sizeof(uint32_t));  // needed??
  memset(clusters_h.view().moduleStart(), 0, (nModules + 1) * sizeof(uint32_t));
  memset(clusters_h.view().moduleId(), 0, (nModules + 1) * sizeof(uint32_t));
  memset(clusters_h.view().clusModuleStart(), 0, (nModules + 1) * sizeof(uint32_t));

  assert(0 == clusters_h.view()[nModules].clusInModule());
  clusters_h.view()[1].moduleStart() = 0;

  // fill cluster arrays
  int numberOfClusters = 0;
  for (auto const& dsv : input) {
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    assert(gind < nModules);
    auto const nclus = dsv.size();
    clusters_h.view()[gind].clusInModule() = nclus;
    numberOfClusters += nclus;
  }
  clusters_h.view()[0].clusModuleStart() = 0;

  for (int i = 1; i < nModules + 1; ++i) {
    clusters_h.view()[i].clusModuleStart() =
        clusters_h.view()[i - 1].clusModuleStart() + clusters_h.view()[i - 1].clusInModule();
  }

  assert((uint32_t)numberOfClusters == clusters_h.view()[nModules].clusModuleStart());
  // output SoA
  // element 96 is the start of BPIX2 (i.e. the number of clusters in BPIX1)
  HitsOnHost output(
      numberOfClusters, clusters_h.view()[startBPIX2].clusModuleStart(), &cpeView, clusters_h.view().clusModuleStart());

  if (0 == numberOfClusters) {
    iEvent.emplace(tokenHit_, std::move(output));
    if (convert2Legacy_)
      iEvent.put(std::move(legacyOutput));
    return;
  }

  if (convert2Legacy_)
    legacyOutput->reserve(nModules, numberOfClusters);

  int numberOfDetUnits = 0;
  int numberOfHits = 0;
  for (auto const& dsv : input) {
    numberOfDetUnits++;
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    assert(gind < nModules);
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    auto const nclus = dsv.size();

    assert(clusters_h.view()[gind].clusInModule() == nclus);
    if (0 == nclus)
      continue;  // is this really possible?

    auto const fc = clusters_h.view()[gind].clusModuleStart();
    auto const lc = clusters_h.view()[gind + 1].clusModuleStart();
    assert(lc > fc);
    LogDebug("SiPixelRecHitSoAFromLegacy") << "in det " << gind << ": conv " << nclus << " hits from " << dsv.size()
                                           << " legacy clusters" << ' ' << fc << ',' << lc;
    assert((lc - fc) == nclus);
    if (nclus > maxHitsInModule)
      printf(
          "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nclus, gind, maxHitsInModule);

    // count digis
    uint32_t ndigi = 0;
    for (auto const& clust : dsv) {
      assert(clust.size() > 0);
      ndigi += clust.size();
    }

    cms::cuda::PortableHostCollection<SiPixelDigisSoALayout<>> digis_h(ndigi);

    clusterRef.clear();
    clusters_h.view()[0].moduleId() = gind;

    uint32_t ic = 0;
    ndigi = 0;
    //filling digis
    for (auto const& clust : dsv) {
      assert(clust.size() > 0);
      for (int i = 0, nd = clust.size(); i < nd; ++i) {
        auto px = clust.pixel(i);
        digis_h.view()[ndigi].xx() = px.x;
        digis_h.view()[ndigi].yy() = px.y;
        digis_h.view()[ndigi].adc() = px.adc;
        digis_h.view()[ndigi].moduleId() = gind;
        digis_h.view()[ndigi].clus() = ic;
        ++ndigi;
      }

      if (convert2Legacy_)
        clusterRef.emplace_back(edmNew::makeRefTo(hclusters, &clust));
      ic++;
    }
    assert(nclus == ic);

    numberOfHits += nclus;
    // filled creates view
    assert(digis_h.view()[0].adc() != 0);
    // we run on blockId.x==0

    gpuPixelRecHits::getHits(&cpeView, &bsHost, digis_h.view(), ndigi, clusters_h.view(), output.view());
    for (auto h = fc; h < lc; ++h)
      if (h - fc < maxHitsInModule)
        assert(gind == output.view()[h].detectorIndex());
      else
        assert(gpuClustering::invalidModuleId == output.view()[h].detectorIndex());

    if (convert2Legacy_) {
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(*legacyOutput, detid);
      for (auto h = fc; h < lc; ++h) {
        auto ih = h - fc;

        if (ih >= maxHitsInModule)
          break;

        assert(ih < clusterRef.size());
        LocalPoint lp(output.view()[h].xLocal(), output.view()[h].yLocal());
        LocalError le(output.view()[h].xerrLocal(), 0, output.view()[h].yerrLocal());

        SiPixelRecHitQuality::QualWordType rqw = 0;
        SiPixelRecHit hit(lp, le, rqw, *genericDet, clusterRef[ih]);
        recHitsOnDetUnit.push_back(hit);
      }
    }
  }

  assert(numberOfHits == numberOfClusters);

  // fill data structure to support CA
  constexpr auto nLayers = TrackerTraits::numberOfLayers;
  for (auto i = 0U; i < nLayers + 1; ++i) {
    output.view().hitsLayerStart()[i] = clusters_h.view()[cpeView.layerGeometry().layerStart[i]].clusModuleStart();
    LogDebug("SiPixelRecHitSoAFromLegacy")
        << "Layer n." << i << " - starting at module: " << cpeView.layerGeometry().layerStart[i]
        << " - starts ad cluster: " << output->hitsLayerStart()[i] << "\n";
  }

  cms::cuda::fillManyFromVector(&(output.view().phiBinner()),
                                nLayers,
                                output.view().iphi(),
                                output.view().hitsLayerStart().data(),
                                output.view().nHits(),
                                256,
                                output.view().phiBinnerStorage());

  LogDebug("SiPixelRecHitSoAFromLegacy") << "created HitSoa for " << numberOfClusters << " clusters in "
                                         << numberOfDetUnits << " Dets"
                                         << "\n";

  // copy pointer to data (SoA view) to allocated buffer
  memcpy(hitsModuleStart, clusters_h.view().clusModuleStart(), nModules * sizeof(uint32_t));

  iEvent.emplace(tokenHit_, std::move(output));
  if (convert2Legacy_)
    iEvent.put(std::move(legacyOutput));
}

using SiPixelRecHitSoAFromLegacyPhase1 = SiPixelRecHitSoAFromLegacyT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase1);

using SiPixelRecHitSoAFromLegacyPhase2 = SiPixelRecHitSoAFromLegacyT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase2);

using SiPixelRecHitSoAFromLegacyHIonPhase1 = SiPixelRecHitSoAFromLegacyT<pixelTopology::HIonPhase1>;
DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyHIonPhase1);
