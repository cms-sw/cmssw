#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <Math/VectorUtil.h>

#include "DataFormats/HGCalReco/interface/TICLGraph.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

using namespace ticl;

class TICLGraphProducer : public edm::stream::EDProducer<> {
public:
  explicit TICLGraphProducer(const edm::ParameterSet &ps);
  ~TICLGraphProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob();
  void endJob();
  bool isPointInCone(const ticl::Trackster::Vector &coneOrigin,
                     const ticl::Trackster::Vector &direction,
                     const float halfAngle,
                     float maxHeight,
                     const ticl::Trackster::Vector &testPoint);

  void beginRun(edm::Run const &iEvent, edm::EventSetup const &es) override;

private:
  const edm::EDGetTokenT<std::vector<Trackster>> tracksters_clue3d_token_;
  // const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layer_clusters_token_;
  // const StringCutObjectSelector<reco::Track> cutTk_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  // const std::string detector_;
  // const std::string propName_;
  // const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  // const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;

  // const HGCalDDDConstants *hgcons_;
  hgcal::RecHitTools rhtools_;
  // edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
  float del_;
  float angle_first_cone_;
  float angle_second_cone_;
  float max_height_cone_;
};

TICLGraphProducer::TICLGraphProducer(const edm::ParameterSet &ps)
    : tracksters_clue3d_token_(consumes<std::vector<Trackster>>(ps.getParameter<edm::InputTag>("trackstersclue3d"))),
      //    tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      layer_clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layer_clusters"))),
      //    cutTk_(ps.getParameter<std::string>("cutTk")),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      //    detector_(ps.getParameter<std::string>("detector")),
      //    propName_(ps.getParameter<std::string>("propagator")),
      //    bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      //    propagator_token_(
      //        esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      del_(ps.getParameter<double>("wind")),
      angle_first_cone_(ps.getParameter<double>("angle1")),
      angle_second_cone_(ps.getParameter<double>("angle2")),
      max_height_cone_(ps.getParameter<double>("maxConeHeight")) {
  produces<TICLGraph>();
  produces<TICLGraph>("cone");
  //  std::string detectorName_ = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
  //  hdc_token_ =
  //      esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", detectorName_));
}

void TICLGraphProducer::beginJob() {}

void TICLGraphProducer::endJob(){};

void TICLGraphProducer::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  // edm::ESHandle<HGCalDDDConstants> hdc = es.getHandle(hdc_token_);
  // hgcons_ = hdc.product();

  edm::ESHandle<CaloGeometry> geom = es.getHandle(geometry_token_);
  rhtools_.setGeometry(*geom);

  // edm::ESHandle<MagneticField> bfield = es.getHandle(bfield_token_);
  // edm::ESHandle<Propagator> propagator = es.getHandle(propagator_token_);
  //std::cout << "Graph window " << del_ << std::endl;
};

bool TICLGraphProducer::isPointInCone(const ticl::Trackster::Vector &coneOrigin,
                                      const ticl::Trackster::Vector &direction,
                                      const float halfAngle,
                                      float maxHeight,
                                      const ticl::Trackster::Vector &testPoint) {
  //std::cout << "Test Point " << testPoint << " coneOrigin " << coneOrigin << std::endl;
  ticl::Trackster::Vector toCheck = testPoint - coneOrigin;
  //std::cout << "To check " << toCheck << std::endl;
  float proj = toCheck.Dot(direction.Unit());
  //std::cout << "Projection " << proj << std::endl;
  if (proj < 0.f || proj > maxHeight) {
    return false;
  }
  //

  //std::cout << "Max Height " << maxHeight << std::endl;
  double angle = ROOT::Math::VectorUtil::Angle(direction, toCheck);

  // Check if the angle is less than halfAngle
  //std::cout << "Angle " << angle << std::endl;
  return angle < halfAngle;
  //  float baseRadius = maxHeight * std::tan(halfAngle);
  //  float coneRadius = (proj / maxHeight) * baseRadius;
  //  float perpDist = std::sqrt((toCheck - proj * direction.Unit()).Mag2());
  //  //std::cout << "Cone origin " << coneOrigin << " Point " << testPoint << " direction " << direction << std::endl;
  //  //std::cout << "Base Radius " << baseRadius << " Radius at proj " << coneRadius << " perpDist " << perpDist
  //            << std::endl;
  //
  //  return perpDist <= coneRadius;
}

void TICLGraphProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  edm::Handle<std::vector<Trackster>> trackstersclue3d_h;
  evt.getByToken(tracksters_clue3d_token_, trackstersclue3d_h);
  auto trackstersclue3d = *trackstersclue3d_h;
  const auto &layerClusters = evt.get(layer_clusters_token_);

  TICLLayerTile tracksterTilePos;
  TICLLayerTile tracksterTileNeg;

  for (size_t id_t = 0; id_t < trackstersclue3d.size(); ++id_t) {
    auto t = trackstersclue3d[id_t];
    if (t.barycenter().eta() > 0.) {
      tracksterTilePos.fill(t.barycenter().eta(), t.barycenter().phi(), id_t);
    } else if (t.barycenter().eta() < 0.) {
      tracksterTileNeg.fill(t.barycenter().eta(), t.barycenter().phi(), id_t);
    }
  }

  auto findPoint =
      [&](float percentage, const float trackster_energy, const std::vector<unsigned int> vertices) -> float {
    std::vector<float> energyInLayer(rhtools_.lastLayer(), 0.);
    std::vector<float> cumulativeEnergyInLayer(rhtools_.lastLayer(), 0.);
    for (auto const &v : vertices) {
      auto const &lc = layerClusters[v];
      auto const &n_lay = rhtools_.getLayerWithOffset(lc.hitsAndFractions()[0].first);
      energyInLayer[n_lay] += lc.energy() / trackster_energy;
    }
    auto sum = 0.;
    for (size_t iC = 0; iC != energyInLayer.size(); iC++) {
      sum += energyInLayer[iC];
      cumulativeEnergyInLayer[iC] = sum;
    }
    auto layerI = std::min_element(
        cumulativeEnergyInLayer.begin(), cumulativeEnergyInLayer.end(), [percentage](float a, float b) {
          // Check if 'a' and 'b' are both greater than 0
          if (a > 0 && b > 0) {
            // Compare based on absolute difference from 'percentage'
            return std::abs(a - percentage) < std::abs(b - percentage);
          } else if (a > 0) {
            // 'a' is greater than 0, so it is the better choice
            return true;
          } else if (b > 0) {
            // 'b' is greater than 0, so it is the better choice
            return false;
          } else {
            // Both 'a' and 'b' are non-positive, prefer 'a'
            return true;
          }
        });
    if (layerI != cumulativeEnergyInLayer.end()) {
      int layer = std::distance(cumulativeEnergyInLayer.begin(), layerI);
      return rhtools_.getPositionLayer(layer, false).z();
    } else {
      return 0.;
    }
  };

  auto intersectLineWithSurface = [](float surfaceZ, const ticl::Trackster::Vector &origin, const ticl::Trackster::Vector &direction) -> ticl::Trackster::Vector {
    auto const t = (surfaceZ - origin.Z()) / direction.Z();
    auto const iX = t * direction.X() + origin.X();
    auto const iY = t * direction.Y() + origin.Y();
    auto const iZ = surfaceZ;

    const ticl::Trackster::Vector intersection(iX, iY, iZ);
    return intersection;
  };

  auto returnSkeletons = [&](const Trackster &trackster) -> std::array<ticl::Trackster::Vector, 3> {
    auto const &vertices = trackster.vertices();
    std::vector<reco::CaloCluster> vertices_lcs(vertices.size());
    std::transform(vertices.begin(), vertices.end(), vertices_lcs.begin(), [&layerClusters](unsigned int index) {
      return layerClusters[index];
    });
    std::sort(vertices_lcs.begin(), vertices_lcs.end(), [](reco::CaloCluster &c1, reco::CaloCluster &c2) {
      return c1.position().z() < c2.position().z();
    });

    auto const firstLayerZ = findPoint(0.1, trackster.raw_energy(), trackster.vertices());
    auto const lastLayerZ = findPoint(0.9, trackster.raw_energy(), trackster.vertices());
    auto const t0_p1 = trackster.barycenter();
    auto const t0_p0 = intersectLineWithSurface(firstLayerZ, t0_p1, trackster.eigenvectors(0));
    auto const t0_p2 = intersectLineWithSurface(lastLayerZ, t0_p1, trackster.eigenvectors(0));
    std::array<ticl::Trackster::Vector, 3> skeleton{{t0_p0, t0_p1, t0_p2}};
    std::sort(skeleton.begin(), skeleton.end(), [](ticl::Trackster::Vector &v1, ticl::Trackster::Vector &v2) { return v1.Z() < v2.Z(); });
    return skeleton;
  };

  std::vector<Node> allNodes;
  std::vector<Node> allNodes2;
  std::vector<int> isRootNodes(trackstersclue3d.size());

  for (size_t id_t = 0; id_t < trackstersclue3d.size(); ++id_t) {
    allNodes.emplace_back(id_t);
  }

  for (size_t id_t = 0; id_t < trackstersclue3d.size(); ++id_t) {
    allNodes2.emplace_back(id_t);
  }

  const float halfAngle1 = angle_first_cone_;
  const float halfAngle2 = angle_second_cone_;
  const float maxHeightCone = max_height_cone_;
  for (size_t id_t = 0; id_t < trackstersclue3d.size(); ++id_t) {
    auto t = trackstersclue3d[id_t];
    auto const &tracksterEigenvec = t.eigenvectors(0);
    auto const &tracksterEigenVal = t.eigenvalues()[0];
    auto const &skeleton = returnSkeletons(t);

    auto bary = t.barycenter();
    const float del = del_;

    float eta_min = std::max(abs(bary.eta()) - del, TileConstants::minEta);
    float eta_max = std::min(abs(bary.eta()) + del, TileConstants::maxEta);

    if (bary.eta() > 0.) {
      std::array<int, 4> search_box =
          tracksterTilePos.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del, bary.phi() + del);
      if (search_box[2] > search_box[3]) {
        search_box[3] += TileConstants::nPhiBins;
      }

      for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
        for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
          auto &neighbours = tracksterTilePos[tracksterTilePos.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
          for (auto n : neighbours) {
            allNodes[id_t].addNeighbour(n);
            allNodes[n].addNeighbour(id_t);
            auto const &trackster2 = trackstersclue3d[n];
            auto const &skeleton2 = returnSkeletons(trackster2);
            // open first cone
            auto const maxHeight = std::sqrt((skeleton[2] - skeleton[0]).Mag2());
            if (isPointInCone(skeleton[0],
                              tracksterEigenvec * tracksterEigenVal,
                              halfAngle1,
                              maxHeight,
                              skeleton2[0])) {  //first cone
              allNodes2[id_t].addNeighbour(n);
              allNodes2[n].addNeighbour(id_t);
            }
            if (isPointInCone(skeleton[2], t.eigenvectors(0), halfAngle2, maxHeightCone, skeleton2[0])) {  //second cone
              allNodes2[id_t].addNeighbour(n);
              allNodes2[n].addNeighbour(id_t);
            }
          }
        }
      }
    }

    else if (bary.eta() < 0.) {
      std::array<int, 4> search_box =
          tracksterTileNeg.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del, bary.phi() + del);
      if (search_box[2] > search_box[3]) {
        search_box[3] += TileConstants::nPhiBins;
      }

      for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
        for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
          auto &neighbours = tracksterTileNeg[tracksterTileNeg.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
          for (auto n : neighbours) {
            allNodes[id_t].addNeighbour(n);
            allNodes[n].addNeighbour(id_t);
            auto const &trackster2 = trackstersclue3d[n];
            auto const &skeleton2 = returnSkeletons(trackster2);
            // open first cone
            auto const maxHeight = std::sqrt((skeleton[2] - skeleton[0]).Mag2());
            if (isPointInCone(skeleton[0],
                              tracksterEigenvec * tracksterEigenVal,
                              halfAngle1,
                              maxHeight,
                              skeleton2[0])) {  //first cone
              allNodes2[id_t].addNeighbour(n);
              allNodes2[n].addNeighbour(id_t);
            }
            if (isPointInCone(skeleton[2], t.eigenvectors(0), halfAngle2, maxHeightCone, skeleton2[0])) {  //second cone
              allNodes2[id_t].addNeighbour(n);
              allNodes2[n].addNeighbour(id_t);
            }
          }
        }
      }
    }
  }

  auto resultGraph = std::make_unique<TICLGraph>(allNodes, isRootNodes);
  auto resultGraphCone = std::make_unique<TICLGraph>(allNodes2, isRootNodes);

  evt.put(std::move(resultGraph));
  evt.put(std::move(resultGraphCone), "cone");
}

void TICLGraphProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("trackstersclue3d", edm::InputTag("ticlTrackstersCLUE3DHigh"));
  //  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  //  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  //  desc.add<std::string>("detector", "HGCAL");
  //  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<double>("wind", 0.7);
  desc.add<double>("angle1", 0.523599);
  desc.add<double>("angle2", 0.349066);
  desc.add<double>("maxConeHeight", 500.);
  //  desc.add<std::string>("cutTk",
  //                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
  //                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  descriptions.add("ticlGraphProducer", desc);
}

DEFINE_FWK_MODULE(TICLGraphProducer);
