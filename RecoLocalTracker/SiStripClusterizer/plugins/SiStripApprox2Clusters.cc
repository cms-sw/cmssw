#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster_v1.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection_v1.h"
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
  edm::EDGetTokenT<v1::SiStripApproximateClusterCollection> clusterToken_v1_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  bool v1;
};

SiStripApprox2Clusters::SiStripApprox2Clusters(const edm::ParameterSet& conf) {
  tkGeomToken_ = esConsumes();
  v1 = conf.getParameter<bool>("v1");
  if (v1) {
    clusterToken_v1_ = consumes(conf.getParameter<edm::InputTag>("inputApproxClusters"));
  } else {
    clusterToken_ = consumes(conf.getParameter<edm::InputTag>("inputApproxClusters"));
  }
  produces<edmNew::DetSetVector<SiStripCluster>>();
}

void SiStripApprox2Clusters::produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& iSetup) const {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto& tkDets = tkGeom->dets();

  if (clusterToken_v1_.isUninitialized()) {
    const auto& clusterCollection = event.get(clusterToken_);
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
  } else {
    const auto& clusterCollection = event.get(clusterToken_v1_);
    std::vector<uint16_t> v_strip;
    float previous_barycenter = -999.;
    unsigned int module_length = 0;
    unsigned int previous_module_length = 0;

    unsigned int clusBegin = 0;
    for (const auto& detClusters : clusterCollection) {
      edmNew::DetSetVector<SiStripCluster>::FastFiller ff{*result, detClusters.id()};
      unsigned int detId = detClusters.id();

      uint16_t nStrips{0};
      auto det = std::find_if(tkDets.begin(), tkDets.end(), [detId](auto& elem) -> bool {
        return (elem->geographicalId().rawId() == detId);
      });
      const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
      nStrips = p.nstrips();

      v_strip.push_back(nStrips);
      previous_module_length += (v_strip.size() < 3) ? 0 : v_strip[v_strip.size() - 3];
      module_length += (v_strip.size() < 2) ? 0 : v_strip[v_strip.size() - 2];
      bool first_cluster = true;
      detClusters.move(clusBegin);

      for (const auto& cluster : detClusters) {
        const auto convertedCluster = SiStripCluster(cluster,
                                                     nStrips - 1,
                                                     previous_barycenter,
                                                     module_length,
                                                     first_cluster ? previous_module_length : module_length);
        if ((convertedCluster.barycenter()) >= nStrips) {
          break;
        }
        previous_barycenter = convertedCluster.barycenter();
        ++clusBegin;
        ff.push_back(convertedCluster);
        first_cluster = false;
      }
    }
  }

  event.put(std::move(result));
}

void SiStripApprox2Clusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputApproxClusters", edm::InputTag("siStripClusters"));
  desc.add<bool>("v1", false);
  descriptions.add("SiStripApprox2Clusters", desc);
}

DEFINE_FWK_MODULE(SiStripApprox2Clusters);
