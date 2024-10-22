#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include "CalibTracker/SiStripCommon/interface/SiStripOnTrackClusterTableProducerBase.h"

class SiStripPositionCorrectionsTableProducer : public SiStripOnTrackClusterTableProducerBase {
public:
  explicit SiStripPositionCorrectionsTableProducer(const edm::ParameterSet& params)
      : SiStripOnTrackClusterTableProducerBase(params),
        m_clusterInfo(consumesCollector()),
        m_tkGeomToken{esConsumes<>()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "cluster");
    desc.add<std::string>("doc", "On-track cluster properties for Lorentz angle and backplane correction measurement");
    desc.add<bool>("extension", false);
    desc.add<edm::InputTag>("Tracks", edm::InputTag{"generalTracks"});
    descriptions.add("siStripPositionCorrectionsTable", desc);
  }

  void fillTable(const std::vector<OnTrackCluster>& clusters,
                 const edm::View<reco::Track>& tracks,
                 nanoaod::FlatTable* table,
                 const edm::EventSetup& iSetup) final;

private:
  SiStripClusterInfo m_clusterInfo;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_tkGeomToken;
};

void SiStripPositionCorrectionsTableProducer::fillTable(const std::vector<OnTrackCluster>& clusters,
                                                        const edm::View<reco::Track>& tracks,
                                                        nanoaod::FlatTable* table,
                                                        const edm::EventSetup& iSetup) {
  const auto& tkGeom = iSetup.getData(m_tkGeomToken);
  std::vector<uint32_t> c_nstrips;
  std::vector<float> c_barycenter, c_variance, c_localdirx, c_localdiry, c_localdirz, c_localx, c_rhlocalx,
      c_rhlocalxerr;
  for (const auto clus : clusters) {
    c_nstrips.push_back(clus.cluster->amplitudes().size());
    m_clusterInfo.setCluster(*clus.cluster, clus.det);
    c_variance.push_back(m_clusterInfo.variance());
    const auto& trajState = clus.measurement.updatedState();
    const auto trackDir = trajState.localDirection();
    c_localdirx.push_back(trackDir.x());
    c_localdiry.push_back(trackDir.y());
    c_localdirz.push_back(trackDir.z());
    const auto hit = clus.measurement.recHit()->hit();
    const auto stripDet = dynamic_cast<const StripGeomDetUnit*>(tkGeom.idToDet(hit->geographicalId()));
    c_barycenter.push_back(stripDet->specificTopology().localPosition(clus.cluster->barycenter()).x());
    c_localx.push_back(stripDet->toLocal(trajState.globalPosition()).x());
    c_rhlocalx.push_back(hit->localPosition().x());
    c_rhlocalxerr.push_back(hit->localPositionError().xx());
  }
  addColumn(table, "nstrips", c_nstrips, "cluster width");
  addColumn(table, "variance", c_variance, "Cluster variance");
  addColumn(table, "localdirx", c_localdirx, "x component of the local track direction");
  addColumn(table, "localdiry", c_localdiry, "y component of the local track direction");
  addColumn(table, "localdirz", c_localdirz, "z component of the local track direction");
  addColumn(table, "barycenter", c_barycenter, "Cluster barycenter (local x without corrections)");
  addColumn(table, "localx", c_localx, "Track local x");
  addColumn(table, "rhlocalx", c_rhlocalx, "RecHit local x");
  addColumn(table, "rhlocalxerr", c_rhlocalxerr, "RecHit local x uncertainty");
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripPositionCorrectionsTableProducer);
