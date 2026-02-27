#include <memory>
#include <unordered_set>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

// local include(s)
#include "PixelClusterizerBase.h"

//#define EDM_ML_DEBUG
//#define GPU_DEBUG

template <typename TrackerTraits>
class SiPixelDigisClustersFromSoAAlpaka : public edm::global::EDProducer<> {
public:
  explicit SiPixelDigisClustersFromSoAAlpaka(const edm::ParameterSet& iConfig);
  ~SiPixelDigisClustersFromSoAAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> const topoToken_;
  edm::EDGetTokenT<SiPixelDigisHost> const digisHostToken_;
  const SiPixelClusterThresholds clusterThresholds_;  // Cluster threshold in electrons
  const bool produceDigis_;
  const bool storeDigis_;

  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> digisPutToken_;
  edm::EDPutTokenT<SiPixelClusterCollectionNew> clustersPutToken_;
};

template <typename TrackerTraits>
SiPixelDigisClustersFromSoAAlpaka<TrackerTraits>::SiPixelDigisClustersFromSoAAlpaka(const edm::ParameterSet& iConfig)
    : topoToken_(esConsumes()),
      digisHostToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      clusterThresholds_(iConfig.getParameter<int>("clusterThreshold_layer1"),
                         iConfig.getParameter<int>("clusterThreshold_otherLayers")),
      produceDigis_(iConfig.getParameter<bool>("produceDigis")),
      storeDigis_(produceDigis_ && iConfig.getParameter<bool>("storeDigis")),
      clustersPutToken_(produces()) {
  if (produceDigis_)
    digisPutToken_ = produces<edm::DetSetVector<PixelDigi>>();
}

template <typename TrackerTraits>
void SiPixelDigisClustersFromSoAAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigisSoA"));
  desc.add<int>("clusterThreshold_layer1", pixelClustering::clusterThresholdLayerOne);
  desc.add<int>("clusterThreshold_otherLayers", pixelClustering::clusterThresholdOtherLayers);
  desc.add<bool>("produceDigis", true);
  desc.add<bool>("storeDigis", true);

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiPixelDigisClustersFromSoAAlpaka<TrackerTraits>::produce(edm::StreamID,
                                                               edm::Event& iEvent,
                                                               const edm::EventSetup& iSetup) const {
  const auto& digisHost = iEvent.get(digisHostToken_);
  const auto& digisView = digisHost.const_view();
  const uint32_t nDigis = digisHost.nDigis();

  const auto& ttopo = iSetup.getData(topoToken_);
  constexpr auto maxModules = TrackerTraits::numberOfModules;

  std::unique_ptr<edm::DetSetVector<PixelDigi>> outputDigis;
  if (produceDigis_)
    outputDigis = std::make_unique<edm::DetSetVector<PixelDigi>>();
  if (storeDigis_)
    outputDigis->reserve(maxModules);
  auto outputClusters = std::make_unique<SiPixelClusterCollectionNew>();
  outputClusters->reserve(maxModules, nDigis / 2);

  edm::DetSet<PixelDigi>* detDigis = nullptr;
  uint32_t detId = 0;

  for (uint32_t i = 0; i < nDigis; i++) {
    // check for uninitialized digis
    // this is set in RawToDigi_kernel in SiPixelRawToClusterGPUKernel.cu
    if (digisView[i].rawIdArr() == 0)
      continue;

    // check for noisy/dead pixels (electrons set to 0)
    if (digisView[i].adc() == 0)
      continue;

    detId = digisView[i].rawIdArr();
    if (storeDigis_) {
      detDigis = &outputDigis->find_or_insert(detId);

      if ((*detDigis).empty())
        (*detDigis).data.reserve(64);  // avoid the first relocations
    }

    break;
  }

  int32_t nclus = -1;
  PixelClusterizerBase::AccretionCluster aclusters[TrackerTraits::maxNumClustersPerModules];
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
        edm::LogWarning("SiPixelDigisClustersFromSoAAlpaka")
            << "cluster below charge Threshold "
            << "Layer/DetId/clusId " << layer << '/' << detId << '/' << ic << " size/charge " << acluster.isize << '/'
            << acluster.charge << "\n";
      // sort by row (x)
      spc.emplace_back(acluster.isize, acluster.adc, acluster.x, acluster.y, acluster.xmin, acluster.ymin, ic);
      aclusters[ic].clear();
#ifdef EDM_ML_DEBUG
      ++totClustersFilled;
      const auto& cluster{spc.back()};
      // LogDebug("SiPixelDigisClustersFromSoAAlpaka")
      std::cout << "putting in this cluster " << ic << " " << cluster.charge() << " " << cluster.pixelADC().size()
                << "\n";
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

#ifdef GPU_DEBUG
  std::cout << "Dumping all digis. nDigis = " << nDigis << std::endl;
#endif
  std::unordered_set<uint32_t> processedDetIds;
  for (uint32_t i = 0; i < nDigis; i++) {
    // check for uninitialized digis
    if (digisView[i].rawIdArr() == 0)
      continue;
    // check for noisy/dead pixels (electrons set to 0)
    if (digisView[i].adc() == 0)
      continue;
    // not in cluster; TODO add an assert for the size
    if (digisView[i].clus() == pixelClustering::invalidClusterId) {
      continue;
    }
    // unexpected invalid value
    if (digisView[i].clus() < pixelClustering::invalidClusterId) {
      edm::LogError("SiPixelDigisClustersFromSoAAlpaka")
          << "Skipping pixel digi with unexpected invalid cluster id " << digisView[i].clus();
      continue;
    }
    // from clusters killed by charge cut
    if (digisView[i].clus() == pixelClustering::invalidModuleId)
      continue;

#ifdef EDM_ML_DEBUG
    assert(digisView[i].rawIdArr() > 109999);
#endif
    // Check if detId has already been processed
    if (processedDetIds.find(detId) != processedDetIds.end()) {
      continue;  // Skip processing this detId if it has already been handled
    }
    if (detId != digisView[i].rawIdArr()) {
#ifdef GPU_DEBUG
      std::cout << ">> Closed module --" << detId << "; nclus = " << nclus << std::endl;
#endif
      // new module
      fillClusters(detId);
      processedDetIds.insert(detId);
#ifdef EDM_ML_DEBUG
      assert(nclus == -1);
#endif
      detId = digisView[i].rawIdArr();
      if (storeDigis_) {
        detDigis = &outputDigis->find_or_insert(detId);
        if ((*detDigis).empty())
          (*detDigis).data.reserve(64);  // avoid the first relocations
        else {
          edm::LogWarning("SiPixelDigisClustersFromSoAAlpaka")
              << "Problem det present twice in input! " << (*detDigis).detId();
        }
      }
    }
    PixelDigi dig{digisView[i].pdigi()};
#ifdef GPU_DEBUG
    std::cout << i << ";" << digisView[i].rawIdArr() << ";" << digisView[i].clus() << ";" << digisView[i].pdigi() << ";"
              << digisView[i].adc() << ";" << dig.row() << ";" << dig.column() << std::endl;
#endif

    if (storeDigis_)
      (*detDigis).data.emplace_back(dig);
      // fill clusters
#ifdef EDM_ML_DEBUG
    assert(digisView[i].clus() >= 0);
    assert(digisView[i].clus() < static_cast<int>(TrackerTraits::maxNumClustersPerModules));
#endif
    nclus = std::max(digisView[i].clus(), nclus);
    auto row = dig.row();
    auto col = dig.column();
    SiPixelCluster::PixelPos pix(row, col);
    aclusters[digisView[i].clus()].add(pix, digisView[i].adc());
  }

  // fill final clusters (if not already filled for that ID)
  if (detId > 0 && processedDetIds.find(detId) == processedDetIds.end())
    fillClusters(detId);

#ifdef EDM_ML_DEBUG
  LogDebug("SiPixelDigisClustersFromSoAAlpaka") << "filled " << totClustersFilled << " clusters";
#endif

  if (produceDigis_)
    iEvent.put(digisPutToken_, std::move(outputDigis));

  iEvent.put(clustersPutToken_, std::move(outputClusters));
}

#include "FWCore/Framework/interface/MakerMacros.h"

using SiPixelDigisClustersFromSoAAlpakaPhase1 = SiPixelDigisClustersFromSoAAlpaka<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelDigisClustersFromSoAAlpakaPhase1);

using SiPixelDigisClustersFromSoAAlpakaPhase2 = SiPixelDigisClustersFromSoAAlpaka<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelDigisClustersFromSoAAlpakaPhase2);

using SiPixelDigisClustersFromSoAAlpakaHIonPhase1 = SiPixelDigisClustersFromSoAAlpaka<pixelTopology::HIonPhase1>;
DEFINE_FWK_MODULE(SiPixelDigisClustersFromSoAAlpakaHIonPhase1);
