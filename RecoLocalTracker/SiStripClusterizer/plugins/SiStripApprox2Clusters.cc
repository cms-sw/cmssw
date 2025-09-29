#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollectionV2.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include <vector>
#include <memory>

class SiStripApprox2Clusters : public edm::global::EDProducer<> {
public:
  explicit SiStripApprox2Clusters(const edm::ParameterSet& conf);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename CollectionType>
  void processCollection(const CollectionType& clusterCollection,
                         edmNew::DetSetVector<SiStripCluster>& result,
                         const edm::EventSetup& iSetup) const;

  unsigned int collectionVersion;
  edm::EDGetTokenT<SiStripApproximateClusterCollection> clusterTokenV1_;
  edm::EDGetTokenT<SiStripApproximateClusterCollectionV2> clusterTokenV2_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  SiStripDetInfo detInfo_;
};

SiStripApprox2Clusters::SiStripApprox2Clusters(const edm::ParameterSet& conf) {
  const auto inputTag = conf.getParameter<edm::InputTag>("inputApproxClusters");
  collectionVersion = conf.getParameter<unsigned int>("collectionVersion");

  clusterTokenV1_ = consumes<SiStripApproximateClusterCollection>(inputTag);
  clusterTokenV2_ = consumes<SiStripApproximateClusterCollectionV2>(inputTag);

  tkGeomToken_ = esConsumes();
  detInfo_ = SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
  produces<edmNew::DetSetVector<SiStripCluster>>();
}

void SiStripApprox2Clusters::produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& iSetup) const {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();

  if (collectionVersion == 1) {
    const auto& clusterCollection = event.get(clusterTokenV1_);
    processCollection(clusterCollection, *result, iSetup);
  } else if (collectionVersion == 2) {
    const auto& clusterCollection = event.get(clusterTokenV2_);
    processCollection(clusterCollection, *result, iSetup);
  } else {
    throw cms::Exception("InvalidParameter")
        << "Invalid collectionVersion: " << collectionVersion << ". Must be 1 or 2.";
  }

  event.put(std::move(result));
}

template <typename CollectionType>
void SiStripApprox2Clusters::processCollection(const CollectionType& clusterCollection,
                                               edmNew::DetSetVector<SiStripCluster>& result,
                                               const edm::EventSetup& iSetup) const {
  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& tkDets = tkGeom->dets();

  float previous_barycenter = SiStripApproximateCluster::barycenterOffset_;
  unsigned int offset_module_change = 0;
  unsigned int clusBegin = 0;

  for (const auto& detClusters : clusterCollection) {
    edmNew::DetSetVector<SiStripCluster>::FastFiller ff{result, detClusters.id()};
    unsigned int detId = detClusters.id();

    uint16_t nStrips{0};
    auto det = std::find_if(tkDets.begin(), tkDets.end(), [detId](auto& elem) -> bool {
      return (elem->geographicalId().rawId() == detId);
    });
    const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
    nStrips = p.nstrips() - 1;

    if constexpr (std::is_same<CollectionType, SiStripApproximateClusterCollectionV2>::value) {
      detClusters.move(clusBegin);
    }
    for (const auto& cluster : detClusters) {
      const auto convertedCluster = SiStripCluster(cluster, nStrips, previous_barycenter, offset_module_change);
      if ((convertedCluster.barycenter()) >= nStrips + 1) {
        // this should not happen for V1
        if (collectionVersion == 1)
          throw cms::Exception("DataCorrupt")
              << "SiStripApprox2Clusters: cluster with barycenter " << convertedCluster.barycenter()
              << " out of range for module with " << nStrips + 1 << " strips.";
        // in V2 is used to split clusters across modules
        break;
      }
      previous_barycenter = convertedCluster.barycenter();
      ++clusBegin;
      offset_module_change = 0;

      ff.push_back(convertedCluster);
    }
    offset_module_change = detInfo_.getNumberOfApvsAndStripLength(detId).first * sistrip::STRIPS_PER_APV;
  }
}

void SiStripApprox2Clusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputApproxClusters", edm::InputTag("siStripClusters"));
  desc.add<unsigned int>(
      "collectionVersion",
      1);  // Collection version (1 for SiStripApproximateClusterCollection, 2 for SiStripApproximateClusterCollectionV2)
  descriptions.add("SiStripApprox2Clusters", desc);
}

DEFINE_FWK_MODULE(SiStripApprox2Clusters);
