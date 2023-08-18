#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
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

#include <vector>
#include <memory>

class SiStripApprox2Clusters : public edm::global::EDProducer<> {
public:
  explicit SiStripApprox2Clusters(const edm::ParameterSet& conf);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<SiStripApproximateClusterCollection> clusterToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
};

SiStripApprox2Clusters::SiStripApprox2Clusters(const edm::ParameterSet& conf) {
  clusterToken_ = consumes(conf.getParameter<edm::InputTag>("inputApproxClusters"));
  tkGeomToken_ = esConsumes();
  produces<edmNew::DetSetVector<SiStripCluster>>();
}

void SiStripApprox2Clusters::produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& iSetup) const {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();
  const auto& clusterCollection = event.get(clusterToken_);

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& tkDets = tkGeom->dets();

  for (const auto& detClusters : clusterCollection) {
    edmNew::DetSetVector<SiStripCluster>::FastFiller ff{*result, detClusters.id()};
    unsigned int detId = detClusters.id();

    uint16_t nStrips{0};
    auto det = std::find_if(tkDets.begin(), tkDets.end(), [detId](auto& elem) -> bool {
      return (elem->geographicalId().rawId() == detId);
    });
    const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
    nStrips = p.nstrips() - 1;

    for (const auto& cluster : detClusters) {
      ff.push_back(SiStripCluster(cluster, nStrips));
    }
  }

  event.put(std::move(result));
}

void SiStripApprox2Clusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputApproxClusters", edm::InputTag("siStripClusters"));
  descriptions.add("SiStripApprox2Clusters", desc);
}

DEFINE_FWK_MODULE(SiStripApprox2Clusters);
