#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/VectorUtil.h>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingbySkeletons.h"
#include "TICLGraph.h"

using namespace ticl;

TracksterLinkingbySkeletons::TracksterLinkingbySkeletons(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TracksterLinkingAlgoBase(conf, iC),
      timing_quality_threshold_(conf.getParameter<double>("track_time_quality_threshold")),
      del_(conf.getParameter<double>("wind")),
      angle_first_cone_(conf.getParameter<double>("angle0")),
      angle_second_cone_(conf.getParameter<double>("angle1")),
      angle_third_cone_(conf.getParameter<double>("angle2")),
      pcaQ_(conf.getParameter<double>("pcaQuality")),
      pcaQLCSize_(conf.getParameter<unsigned int>("pcaQualityLCSize")),
      dotCut_(conf.getParameter<double>("dotProdCut")),
      maxDistSkeletonsSq_(conf.getParameter<double>("maxDistSkeletonsSq")),
      max_height_cone_(conf.getParameter<double>("maxConeHeight")) {}

void TracksterLinkingbySkeletons::buildLayers() {
  // build disks at HGCal front & EM-Had interface for track propagation

  float zVal = hgcons_->waferZ(1, true);
  std::pair<float, float> rMinMax = hgcons_->rangeR(zVal, true);

  float zVal_interface = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
  std::pair<float, float> rMinMax_interface = hgcons_->rangeR(zVal_interface, true);

  for (int iSide = 0; iSide < 2; ++iSide) {
    float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
    firstDisk_[iSide] =
        std::make_unique<GeomDet>(Disk::build(Disk::PositionType(0, 0, zSide),
                                              Disk::RotationType(),
                                              SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                                      .get());

    zSide = (iSide == 0) ? (-1. * zVal_interface) : zVal_interface;
    interfaceDisk_[iSide] = std::make_unique<GeomDet>(
        Disk::build(Disk::PositionType(0, 0, zSide),
                    Disk::RotationType(),
                    SimpleDiskBounds(rMinMax_interface.first, rMinMax_interface.second, zSide - 0.5, zSide + 0.5))
            .get());
  }
}

void TracksterLinkingbySkeletons::initialize(const HGCalDDDConstants *hgcons,
                                             const hgcal::RecHitTools rhtools,
                                             const edm::ESHandle<MagneticField> bfieldH,
                                             const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  buildLayers();

  bfield_ = bfieldH;
  propagator_ = propH;
}

float TracksterLinkingbySkeletons::findSkeletonPoints(float percentage,
                                                      const float trackster_energy,
                                                      const std::vector<unsigned int> vertices,
                                                      const hgcal::RecHitTools &rhtools,
                                                      const std::vector<reco::CaloCluster> &layerClusters) {
  std::vector<float> energyInLayer(rhtools_.lastLayer(), 0.);
  std::vector<float> cumulativeEnergyInLayer(rhtools_.lastLayer(), 0.);
  for (auto const &v : vertices) {
    auto const &lc = layerClusters[v];
    auto const &n_lay = rhtools_.getLayerWithOffset(lc.hitsAndFractions()[0].first);
    energyInLayer[n_lay] += lc.energy() / trackster_energy;
  }
  if (TracksterLinkingAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced) {
    for (size_t iLay = 0; iLay < energyInLayer.size(); ++iLay) {
      LogDebug("TracksterLinkingbySkeletons")
          << "Layer " << iLay << " contains a Trackster energy fraction of " << energyInLayer[iLay]
          << " and the trackster energy is " << trackster_energy << "\n";
    }
  }
  auto sum = 0.;
  for (size_t iC = 0; iC != energyInLayer.size(); iC++) {
    sum += energyInLayer[iC];
    cumulativeEnergyInLayer[iC] = sum;
  }
  if (TracksterLinkingAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced) {
    for (size_t iLay = 0; iLay < cumulativeEnergyInLayer.size(); ++iLay) {
      LogDebug("TracksterLinkingbySkeletons")
          << "Layer " << iLay << " cumulative has a Trackster energy fraction of " << cumulativeEnergyInLayer[iLay]
          << " and the trackster energy is " << trackster_energy << "\n";
    }
  }
  auto layerI =
      std::min_element(cumulativeEnergyInLayer.begin(), cumulativeEnergyInLayer.end(), [percentage](float a, float b) {
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
    if (TracksterLinkingAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced) {
      LogDebug("TracksterLinkingbySkeletons")
          << "Layer containing at least an energy fraction " << percentage << " is " << layer << "\n";
    }
    return rhtools_.getPositionLayer(layer, false).z();
  } else {
    return 0.;
  }
}

void TracksterLinkingbySkeletons::linkTracksters(
    const Inputs &input,
    std::vector<Trackster> &resultTracksters,
    std::vector<std::vector<unsigned int>> &linkedResultTracksters,
    std::vector<std::vector<unsigned int>> &linkedTracksterIdToInputTracksterId) {
  const auto &tracksters = input.tracksters;
  const auto &layerClusters = input.layerClusters;

  // linking : trackster is hadr nic if its barycenter is in CE-H
  auto isHadron = [&](const Trackster &t) -> bool {
    auto boundary_z = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();

    return (std::abs(t.barycenter().Z()) > boundary_z);
  };

  if (TracksterLinkingAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced)
    LogDebug("TracksterLinkingbySkeletons") << "------- Graph Linking ------- \n";

  auto intersectLineWithSurface = [](float surfaceZ, const Vector &origin, const Vector &direction) -> Vector {
    auto const t = (surfaceZ - origin.Z()) / direction.Z();
    auto const iX = t * direction.X() + origin.X();
    auto const iY = t * direction.Y() + origin.Y();
    auto const iZ = surfaceZ;
    const Vector intersection(iX, iY, iZ);
    return intersection;
  };

  auto returnSkeletons = [&](const Trackster &trackster) -> std::array<Vector, 3> {
    auto const &vertices = trackster.vertices();
    std::vector<reco::CaloCluster> vertices_lcs(vertices.size());
    std::transform(vertices.begin(), vertices.end(), vertices_lcs.begin(), [&layerClusters](unsigned int index) {
      return layerClusters[index];
    });

    std::sort(vertices_lcs.begin(), vertices_lcs.end(), [](reco::CaloCluster &c1, reco::CaloCluster &c2) {
      return c1.position().z() < c2.position().z();
    });

    auto const firstLayerZ =
        findSkeletonPoints(0.1f, trackster.raw_energy(), trackster.vertices(), rhtools_, layerClusters);
    auto const lastLayerZ =
        findSkeletonPoints(0.9f, trackster.raw_energy(), trackster.vertices(), rhtools_, layerClusters);
    auto const t0_p1 = trackster.barycenter();
    auto const t0_p0 = intersectLineWithSurface(firstLayerZ, t0_p1, trackster.eigenvectors(0));
    auto const t0_p2 = intersectLineWithSurface(lastLayerZ, t0_p1, trackster.eigenvectors(0));
    std::array<Vector, 3> skeleton{{t0_p0, t0_p1, t0_p2}};
    std::sort(skeleton.begin(), skeleton.end(), [](Vector &v1, Vector &v2) { return v1.Z() < v2.Z(); });
    return skeleton;
  };

  //  auto argSortTrackster = [](const std::vector<Trackster> &tracksters) -> std::vector<unsigned int> {
  //    std::vector<unsigned int> retIndices(tracksters.size());
  //    std::iota(retIndices.begin(), retIndices.end(), 0);
  //    std::stable_sort(retIndices.begin(), retIndices.end(), [&tracksters](unsigned int i, unsigned int j) {
  //      return tracksters[i].raw_energy() > tracksters[j].raw_energy();
  //    });
  //
  //    return retIndices;
  //  };

  auto isPointInCone = [](const Vector &origin,
                          const Vector &direction,
                          const float halfAngle,
                          const float maxHeight,
                          const Vector &testPoint,
                          std::string str = "") -> bool {
    Vector toCheck = testPoint - origin;
    auto projection = toCheck.Dot(direction.Unit());
    auto const angle = ROOT::Math::VectorUtil::Angle(direction, toCheck);
    LogDebug("TracksterLinkingbySkeletons")
        << str << "Origin " << origin << " TestPoint " << testPoint << " projection " << projection << " maxHeight "
        << maxHeight << " Angle " << angle << " halfAngle " << halfAngle << std::endl;
    if (projection < 0.f || projection > maxHeight) {
      return false;
    }

    return angle < halfAngle;
  };

  TICLLayerTile tracksterTilePos;
  TICLLayerTile tracksterTileNeg;

  for (size_t id_t = 0; id_t < tracksters.size(); ++id_t) {
    auto t = tracksters[id_t];
    if (t.barycenter().eta() > 0.) {
      tracksterTilePos.fill(t.barycenter().eta(), t.barycenter().phi(), id_t);
    } else if (t.barycenter().eta() < 0.) {
      tracksterTileNeg.fill(t.barycenter().eta(), t.barycenter().phi(), id_t);
    }
  }

  //actual trackster-trackster linking
  auto pcaQuality = [](const Trackster &trackster) -> float {
    auto const &eigenvalues = trackster.eigenvalues();
    auto const e0 = eigenvalues[0];
    auto const e1 = eigenvalues[1];
    auto const e2 = eigenvalues[2];
    auto const sum = e0 + e1 + e2;
    auto const normalized_e0 = e0 / sum;
    auto const normalized_e1 = e1 / sum;
    auto const normalized_e2 = e2 / sum;
    return normalized_e0;
  };

  const float halfAngle0 = angle_first_cone_;
  const float halfAngle1 = angle_second_cone_;
  const float halfAngle2 = angle_third_cone_;
  const float maxHeightCone = max_height_cone_;

  std::vector<int> maskReceivedLink(tracksters.size(), 1);
  std::vector<int> isRootTracksters(tracksters.size(), 1);

  std::vector<Node> allNodes;
  for (size_t it = 0; it < tracksters.size(); ++it) {
    allNodes.emplace_back(it);
  }

  for (size_t it = 0; it < tracksters.size(); ++it) {
    auto const &trackster = tracksters[it];
    isHadron(trackster);

    auto pcaQ = pcaQuality(trackster);
    LogDebug("TracksterLinkingbySkeletons")
        << "DEBUG Trackster " << it << " energy " << trackster.raw_energy() << " Num verties "
        << trackster.vertices().size() << " PCA Quality " << pcaQ << std::endl;
    const float pcaQTh = pcaQ_;
    const unsigned int pcaQLCSize = pcaQLCSize_;
    if (pcaQ >= pcaQTh && trackster.vertices().size() > pcaQLCSize) {
      auto const skeletons = returnSkeletons(trackster);
      LogDebug("TracksterLinkingbySkeletons")
          << "Trackster " << it << " energy " << trackster.raw_energy() << " Num verties "
          << trackster.vertices().size() << " PCA Quality " << pcaQ << " Skeletons " << skeletons[0] << std::endl;
      auto const &eigenVec = trackster.eigenvectors(0);
      auto const eigenVal = trackster.eigenvalues()[0];
      auto const &directionOrigin = eigenVec * eigenVal;

      auto const bary = trackster.barycenter();
      float eta_min = std::max(abs(bary.eta()) - del_, TileConstants::minEta);
      float eta_max = std::min(abs(bary.eta()) + del_, TileConstants::maxEta);

      if (bary.eta() > 0.) {
        std::array<int, 4> search_box =
            tracksterTilePos.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del_, bary.phi() + del_);
        if (search_box[2] > search_box[3]) {
          search_box[3] += TileConstants::nPhiBins;
        }

        for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
          for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
            auto &neighbours = tracksterTilePos[tracksterTilePos.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
            for (auto n : neighbours) {
              if (maskReceivedLink[n] == 0)
                continue;

              auto const &trackster_out = tracksters[n];
              auto const &skeletons_out = returnSkeletons(trackster_out);
              auto const skeletonDist2 = (skeletons[2] - skeletons_out[0]).Mag2();
              auto const pcaQOuter = pcaQuality(trackster_out);
              //              auto const dotProd  = ((skeletons[0]-skeletons[2]).Unit()).Dot((skeletons_out[0] - skeletons_out[2]).Unit());
              bool isGoodPCA = (pcaQOuter >= pcaQTh) && (trackster_out.vertices().size() > pcaQLCSize);
              auto const maxHeightSmallCone = std::sqrt((skeletons[2] - skeletons[0]).Mag2());
              bool isInSmallCone = isPointInCone(
                  skeletons[0], directionOrigin, halfAngle0, maxHeightSmallCone, skeletons_out[0], "Small Cone");
              bool isInCone =
                  isPointInCone(skeletons[1], directionOrigin, halfAngle1, maxHeightCone, skeletons_out[0], "BigCone ");
              bool isInLastCone =
                  isPointInCone(skeletons[2], directionOrigin, halfAngle2, maxHeightCone, skeletons_out[0], "LastCone");
              bool dotProd =
                  isGoodPCA
                      ? ((skeletons[0] - skeletons[2]).Unit()).Dot((skeletons_out[0] - skeletons_out[2]).Unit()) >=
                            dotCut_
                      : true;

              LogDebug("TracksterLinkingbySkeletons")
                  << "\tTrying to Link Trackster " << n << " energy " << tracksters[n].raw_energy() << " LCs "
                  << tracksters[n].vertices().size() << " skeletons " << skeletons_out[0] << " Dist " << skeletonDist2
                  << " dot Prod "
                  << ((skeletons[0] - skeletons[2]).Unit()).Dot((skeletons_out[0] - skeletons_out[2]).Unit())
                  << " isGoodDotProd " << dotProd << " isPointInBigCone " << isInCone << " isPointInSmallCone "
                  << isInSmallCone << " isPointInLastCone " << isInLastCone << std::endl;
              if (isInLastCone && dotProd) {
                LogDebug("TracksterLinkingbySkeletons") << "\t==== LINK: Trackster " << it << " Linked with Trackster "
                                                        << n << " LCs " << tracksters[n].vertices().size() << std::endl;
                LogDebug("TracksterLinkingbySkeletons")
                    << "\t\tSkeleton origin " << skeletons[2] << " Skeleton out " << skeletons_out[0] << std::endl;
                maskReceivedLink[n] = 0;
                allNodes[it].addNeighbour(n);
                isRootTracksters[n] = 0;
              }
              if (isInCone && skeletonDist2 <= maxDistSkeletonsSq_ && dotProd) {
                LogDebug("TracksterLinkingbySkeletons") << "\t==== LINK: Trackster " << it << " Linked with Trackster "
                                                        << n << " LCs " << tracksters[n].vertices().size() << std::endl;
                LogDebug("TracksterLinkingbySkeletons")
                    << "\t\tSkeleton origin " << skeletons[2] << " Skeleton out " << skeletons_out[0] << std::endl;
                maskReceivedLink[n] = 0;
                allNodes[it].addNeighbour(n);
                isRootTracksters[n] = 0;
                continue;
              }
              if (isInSmallCone && !isGoodPCA) {
                maskReceivedLink[n] = 0;
                allNodes[it].addNeighbour(n);
                isRootTracksters[n] = 0;
                LogDebug("TracksterLinkingbySkeletons")
                    << "\t==== LINK: Trackster " << it << " Linked with Trackster in small cone " << n << std::endl;
                LogDebug("TracksterLinkingbySkeletons")
                    << "\t\tSkeleton origin " << skeletons[0] << " Skeleton out " << skeletons_out[0] << std::endl;
                continue;
              }
            }
          }
        }
      }
    }
  }

  LogDebug("TracksterLinkingbySkeletons") << "****************  FINAL GRAPH **********************" << std::endl;
  for (auto const &node : allNodes) {
    if (isRootTracksters[node.getId()]) {
      LogDebug("TracksterLinkingbySkeletons")
          << "ISROOT "
          << " Node " << node.getId() << " position " << tracksters[node.getId()].barycenter() << " energy "
          << tracksters[node.getId()].raw_energy() << std::endl;
    } else {
      LogDebug("TracksterLinkingbySkeletons")
          << "Node " << node.getId() << " position " << tracksters[node.getId()].barycenter() << " energy "
          << tracksters[node.getId()].raw_energy() << std::endl;
    }
  }
  LogDebug("TracksterLinkingbySkeletons") << "********************************************************" << std::endl;

  TICLGraph graph(allNodes, isRootTracksters);

  int ic = 0;
  auto const &components = graph.findSubComponents();
  linkedTracksterIdToInputTracksterId.resize(components.size());
  for (auto const &comp : components) {
    LogDebug("TracksterLinkingbySkeletons") << "Component " << ic << " Node: ";
    std::vector<unsigned int> linkedTracksters;
    Trackster outTrackster;
    for (auto const &node : comp) {
      LogDebug("TracksterLinkingbySkeletons") << node << " ";
      linkedTracksterIdToInputTracksterId[ic].push_back(node);
      outTrackster.mergeTracksters(input.tracksters[node]);
    }
    linkedTracksters.push_back(resultTracksters.size());
    resultTracksters.push_back(outTrackster);
    linkedResultTracksters.push_back(linkedTracksters);
    LogDebug("TracksterLinkingbySkeletons") << "\n";
    ++ic;
  }
  LogDebug("TracksterLinkingbySkeletons") << "resultLinked " << std::endl;
}  // linkTracksters
