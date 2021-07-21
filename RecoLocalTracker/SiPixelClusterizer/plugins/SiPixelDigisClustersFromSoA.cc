#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// local include(s)
#include "PixelClusterizerBase.h"
#include "SiPixelClusterThresholds.h"

class SiPixelDigisClustersFromSoA : public edm::global::EDProducer<> {
public:
  explicit SiPixelDigisClustersFromSoA(const edm::ParameterSet& iConfig);
  ~SiPixelDigisClustersFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  edm::EDGetTokenT<SiPixelDigisSoA> digiGetToken_;

  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> digiPutToken_;
  edm::EDPutTokenT<SiPixelClusterCollectionNew> clusterPutToken_;

  const SiPixelClusterThresholds clusterThresholds_;  // Cluster threshold in electrons

  const bool produceDigis_;
  const bool storeDigis_;
};

SiPixelDigisClustersFromSoA::SiPixelDigisClustersFromSoA(const edm::ParameterSet& iConfig)
    : topoToken_(esConsumes()),
      digiGetToken_(consumes<SiPixelDigisSoA>(iConfig.getParameter<edm::InputTag>("src"))),
      clusterPutToken_(produces<SiPixelClusterCollectionNew>()),
      clusterThresholds_{iConfig.getParameter<int>("clusterThreshold_layer1"),
                         iConfig.getParameter<int>("clusterThreshold_otherLayers")},
      produceDigis_(iConfig.getParameter<bool>("produceDigis")),
      storeDigis_(iConfig.getParameter<bool>("produceDigis") & iConfig.getParameter<bool>("storeDigis")) {
  if (produceDigis_)
    digiPutToken_ = produces<edm::DetSetVector<PixelDigi>>();
}

void SiPixelDigisClustersFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigisSoA"));
  desc.add<int>("clusterThreshold_layer1", kSiPixelClusterThresholdsDefaultPhase1.layer1);
  desc.add<int>("clusterThreshold_otherLayers", kSiPixelClusterThresholdsDefaultPhase1.otherLayers);
  desc.add<bool>("produceDigis", true);
  desc.add<bool>("storeDigis", true);
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigisClustersFromSoA::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& digis = iEvent.get(digiGetToken_);
  const uint32_t nDigis = digis.size();
  const auto& ttopo = iSetup.getData(topoToken_);

  std::unique_ptr<edm::DetSetVector<PixelDigi>> collection;
  if (produceDigis_)
    collection = std::make_unique<edm::DetSetVector<PixelDigi>>();
  if (storeDigis_)
    collection->reserve(gpuClustering::maxNumModules);
  auto outputClusters = std::make_unique<SiPixelClusterCollectionNew>();
  outputClusters->reserve(gpuClustering::maxNumModules, nDigis / 2);

  edm::DetSet<PixelDigi>* detDigis = nullptr;
  uint32_t detId = 0;
  for (uint32_t i = 0; i < nDigis; i++) {
    if (digis.pdigi(i) == 0)
      continue;
    detId = digis.rawIdArr(i);
    if (storeDigis_) {
      detDigis = &collection->find_or_insert(detId);
      if ((*detDigis).empty())
        (*detDigis).data.reserve(64);  // avoid the first relocations
    }
    break;
  }

  int32_t nclus = -1;
  PixelClusterizerBase::AccretionCluster aclusters[gpuClustering::maxNumClustersPerModules];
#ifdef EDM_ML_DEBUG
  auto totClustersFilled = 0;
#endif

  auto fillClusters = [&](uint32_t detId) {
    if (nclus < 0)
      return;  // this in reality should never happen
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(*outputClusters, detId);
    auto layer = (DetId(detId).subdetId() == 1) ? ttopo.pxbLayer(detId) : 0;
    auto clusterThreshold = clusterThresholds_.getThresholdForLayerOnCondition(layer == 1);
    for (int32_t ic = 0; ic < nclus + 1; ++ic) {
      auto const& acluster = aclusters[ic];
      // in any case we cannot  go out of sync with gpu...
      if (acluster.charge < clusterThreshold)
        edm::LogWarning("SiPixelDigisClustersFromSoA") << "cluster below charge Threshold "
                                                       << "Layer/DetId/clusId " << layer << '/' << detId << '/' << ic
                                                       << " size/charge " << acluster.isize << '/' << acluster.charge;
      // sort by row (x)
      spc.emplace_back(acluster.isize, acluster.adc, acluster.x, acluster.y, acluster.xmin, acluster.ymin, ic);
      aclusters[ic].clear();
#ifdef EDM_ML_DEBUG
      ++totClustersFilled;
      const auto& cluster{spc.back()};
      LogDebug("SiPixelDigisClustersFromSoA")
          << "putting in this cluster " << ic << " " << cluster.charge() << " " << cluster.pixelADC().size();
#endif
      std::push_heap(spc.begin(), spc.end(), [](SiPixelCluster const& cl1, SiPixelCluster const& cl2) {
        return cl1.minPixelRow() < cl2.minPixelRow();
      });
    }
    nclus = -1;
    // sort by row (x)
    std::sort_heap(spc.begin(), spc.end(), [](SiPixelCluster const& cl1, SiPixelCluster const& cl2) {
      return cl1.minPixelRow() < cl2.minPixelRow();
    });
    if (spc.empty())
      spc.abort();
  };

  for (uint32_t i = 0; i < nDigis; i++) {
    if (digis.pdigi(i) == 0)
      continue;
    if (digis.clus(i) > 9000)
      continue;  // not in cluster; TODO add an assert for the size
#ifdef EDM_ML_DEBUG
    assert(digis.rawIdArr(i) > 109999);
#endif
    if (detId != digis.rawIdArr(i)) {
      // new module
      fillClusters(detId);
#ifdef EDM_ML_DEBUG
      assert(nclus == -1);
#endif
      detId = digis.rawIdArr(i);
      if (storeDigis_) {
        detDigis = &collection->find_or_insert(detId);
        if ((*detDigis).empty())
          (*detDigis).data.reserve(64);  // avoid the first relocations
        else {
          edm::LogWarning("SiPixelDigisClustersFromSoA")
              << "Problem det present twice in input! " << (*detDigis).detId();
        }
      }
    }
    PixelDigi dig(digis.pdigi(i));
    if (storeDigis_)
      (*detDigis).data.emplace_back(dig);
      // fill clusters
#ifdef EDM_ML_DEBUG
    assert(digis.clus(i) >= 0);
    assert(digis.clus(i) < gpuClustering::maxNumClustersPerModules);
#endif
    nclus = std::max(digis.clus(i), nclus);
    auto row = dig.row();
    auto col = dig.column();
    SiPixelCluster::PixelPos pix(row, col);
    aclusters[digis.clus(i)].add(pix, digis.adc(i));
  }

  // fill final clusters
  if (detId > 0)
    fillClusters(detId);

#ifdef EDM_ML_DEBUG
  LogDebug("SiPixelDigisClustersFromSoA") << "filled " << totClustersFilled << " clusters";
#endif

  if (produceDigis_)
    iEvent.put(digiPutToken_, std::move(collection));
  iEvent.put(clusterPutToken_, std::move(outputClusters));
}

DEFINE_FWK_MODULE(SiPixelDigisClustersFromSoA);
