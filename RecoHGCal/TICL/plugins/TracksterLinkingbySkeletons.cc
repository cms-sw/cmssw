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

namespace {
  bool isRoundTrackster(std::array<ticl::Vector, 3> skeleton) { return (skeleton[0].Z() == skeleton[2].Z()); }

  bool isGoodTrackster(const ticl::Trackster &trackster,
                       const std::array<ticl::Vector, 3> &skeleton,
                       const unsigned int min_num_lcs,
                       const float min_trackster_energy,
                       const float pca_quality_th) {
    bool isGood = false;

    if (isRoundTrackster(skeleton) or trackster.vertices().size() < min_num_lcs or
        trackster.raw_energy() < min_trackster_energy) {
      isGood = false;
    } else {
      auto const &eigenvalues = trackster.eigenvalues();
      auto const sum = std::accumulate(std::begin(eigenvalues), std::end(eigenvalues), 0.f);
      float pcaQuality = eigenvalues[0] / sum;
      if (pcaQuality > pca_quality_th) {
        isGood = true;
      }
    }
    return isGood;
  }

  //distance between skeletons
  float projective_distance(const ticl::Vector &point1, const ticl::Vector &point2) {
    // squared projective distance
    float r1 = std::sqrt(point1.x() * point1.x() + point1.y() * point1.y());
    float r2_at_z1 =
        std::sqrt(point2.x() * point2.x() + point2.y() * point2.y()) * std::abs(point1.z()) / std::abs(point2.z());
    float delta_phi = reco::deltaPhi(point1.Phi(), point2.Phi());
    float projective_distance = (r1 - r2_at_z1) * (r1 - r2_at_z1) + r2_at_z1 * r2_at_z1 * delta_phi * delta_phi;
    LogDebug("TracksterLinkingbySkeletons") << "Computing distance between point : " << point1 << " And point "
                                            << point2 << " Distance " << projective_distance << std::endl;
    return projective_distance;
  }
}  // namespace

using namespace ticl;

TracksterLinkingbySkeletons::TracksterLinkingbySkeletons(const edm::ParameterSet &conf,
                                                         edm::ConsumesCollector iC,
                                                         cms::Ort::ONNXRuntime const *onnxRuntime)
    : TracksterLinkingAlgoBase(conf, iC),
      timing_quality_threshold_(conf.getParameter<double>("track_time_quality_threshold")),
      del_(conf.getParameter<double>("wind")),
      min_num_lcs_(conf.getParameter<unsigned int>("min_num_lcs")),
      min_trackster_energy_(conf.getParameter<double>("min_trackster_energy")),
      pca_quality_th_(conf.getParameter<double>("pca_quality_th")),
      dot_prod_th_(conf.getParameter<double>("dot_prod_th")),
      max_distance_projective_sqr_(conf.getParameter<std::vector<double>>("max_distance_projective_sqr")),
      min_distance_z_(conf.getParameter<std::vector<double>>("min_distance_z")),
      max_distance_projective_sqr_closest_points_(
          conf.getParameter<std::vector<double>>("max_distance_projective_sqr_closest_points")),
      max_z_distance_closest_points_(conf.getParameter<std::vector<double>>("max_z_distance_closest_points")),
      cylinder_radius_sqr_(conf.getParameter<std::vector<double>>("cylinder_radius_sqr"))

{}

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

std::array<ticl::Vector, 3> TracksterLinkingbySkeletons::findSkeletonNodes(
    const ticl::Trackster &trackster,
    float lower_percentage,
    float upper_percentage,
    const std::vector<reco::CaloCluster> &layerClusters,
    const hgcal::RecHitTools &rhtools) {
  auto const &vertices = trackster.vertices();
  auto const trackster_raw_energy = trackster.raw_energy();
  // sort vertices by layerId
  std::vector<unsigned int> sortedVertices(vertices);
  std::sort(sortedVertices.begin(), sortedVertices.end(), [&layerClusters](unsigned int i, unsigned int j) {
    return std::abs(layerClusters[i].z()) < std::abs(layerClusters[j].z());
  });

  // now loop over sortedVertices and find the layerId that contains the lower_percentage of the energy
  // and the layerId that contains the upper_percentage of the energy
  float cumulativeEnergyFraction = 0.f;
  int innerLayerId = rhtools.getLayerWithOffset(layerClusters[sortedVertices[0]].hitsAndFractions()[0].first);
  float innerLayerZ = layerClusters[sortedVertices[0]].z();
  int outerLayerId = rhtools.getLayerWithOffset(layerClusters[sortedVertices.back()].hitsAndFractions()[0].first);
  float outerLayerZ = layerClusters[sortedVertices.back()].z();
  bool foundInnerLayer = false;
  bool foundOuterLayer = false;
  for (auto const &v : sortedVertices) {
    auto const &lc = layerClusters[v];
    auto const &n_lay = rhtools.getLayerWithOffset(lc.hitsAndFractions()[0].first);
    cumulativeEnergyFraction += lc.energy() / trackster_raw_energy;
    if (cumulativeEnergyFraction >= lower_percentage and not foundInnerLayer) {
      innerLayerId = n_lay;
      innerLayerZ = lc.z();
      foundInnerLayer = true;
    }
    if (cumulativeEnergyFraction >= upper_percentage and not foundOuterLayer) {
      outerLayerId = n_lay;
      outerLayerZ = lc.z();
      foundOuterLayer = true;
    }
  }
  std::array<ticl::Vector, 3> skeleton;
  int minimumDistanceInLayers = 4;
  if (outerLayerId - innerLayerId < minimumDistanceInLayers) {
    skeleton = {{trackster.barycenter(), trackster.barycenter(), trackster.barycenter()}};
  } else {
    auto intersectLineWithSurface = [](float surfaceZ, const Vector &origin, const Vector &direction) -> Vector {
      auto const t = (surfaceZ - origin.Z()) / direction.Z();
      auto const iX = t * direction.X() + origin.X();
      auto const iY = t * direction.Y() + origin.Y();
      auto const iZ = surfaceZ;
      const Vector intersection(iX, iY, iZ);
      return intersection;
    };

    auto const &t0_p1 = trackster.barycenter();
    auto const t0_p0 = intersectLineWithSurface(innerLayerZ, t0_p1, trackster.eigenvectors(0));
    auto const t0_p2 = intersectLineWithSurface(outerLayerZ, t0_p1, trackster.eigenvectors(0));
    skeleton = {{t0_p0, t0_p1, t0_p2}};
    std::sort(skeleton.begin(), skeleton.end(), [](Vector &v1, Vector &v2) { return v1.Z() < v2.Z(); });
  }

  return skeleton;
}

bool isInCylinder(const std::array<ticl::Vector, 3> &mySkeleton,
                  const std::array<ticl::Vector, 3> &otherSkeleton,
                  const float radius_sqr) {
  const auto &first = mySkeleton[0];
  const auto &last = mySkeleton[2];
  const auto &pointToCheck = otherSkeleton[0];

  const auto &cylAxis = last - first;
  const auto &vecToPoint = pointToCheck - first;

  auto axisNorm = cylAxis.Dot(cylAxis);
  auto projLength = vecToPoint.Dot(cylAxis) / axisNorm;
  bool isWithinLength = projLength >= 0 && projLength <= 1;

  if (!isWithinLength)
    return false;

  const auto &proj = cylAxis * projLength;

  const auto &pointOnAxis = first + proj;

  const auto &distance = pointToCheck - pointOnAxis;
  auto distance2 = distance.Dot(distance);

  bool isWithinRadius = distance2 <= radius_sqr;

  return isWithinRadius;
}

bool TracksterLinkingbySkeletons::areCompatible(const ticl::Trackster &myTrackster,
                                                const ticl::Trackster &otherTrackster,
                                                const std::array<ticl::Vector, 3> &mySkeleton,
                                                const std::array<ticl::Vector, 3> &otherSkeleton) {
  //do not start links from small/bad tracksters
  float zVal_interface = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
  if (!isGoodTrackster(myTrackster, mySkeleton, min_num_lcs_, min_trackster_energy_, pca_quality_th_)) {
    LogDebug("TracksterLinkingbySkeletons") << "Inner Trackster with energy " << myTrackster.raw_energy() << " Num LCs "
                                            << myTrackster.vertices().size() << " NOT GOOD " << std::endl;
    return false;
  } else {
    LogDebug("TracksterLinkingbySkeletons") << "Inner Trackster wi energy " << myTrackster.raw_energy() << " Num LCs "
                                            << myTrackster.vertices().size() << " IS GOOD " << std::endl;
    float proj_distance = projective_distance(mySkeleton[1], otherSkeleton[1]);
    auto isEE = mySkeleton[1].z() <= zVal_interface ? 0 : 1;
    bool areAlignedInProjectiveSpace = proj_distance < max_distance_projective_sqr_[isEE];
    LogDebug("TracksterLinkingbySkeletons")
        << "\t Trying to compare with outer Trackster with energy " << otherTrackster.raw_energy() << " Num LCS "
        << otherTrackster.vertices().size() << " Projective distance " << proj_distance << " areAlignedProjective "
        << areAlignedInProjectiveSpace << " TH " << max_distance_projective_sqr_[isEE] << std::endl;
    //check if otherTrackster is good
    if (isGoodTrackster(otherTrackster, otherSkeleton, min_num_lcs_, min_trackster_energy_, pca_quality_th_)) {
      // if both tracksters are good, then we can check the projective distance between the barycenters.
      // if the barycenters are aligned, then we check that the two skeletons are aligned
      if (areAlignedInProjectiveSpace) {
        auto dotProdSkeletons =
            ((mySkeleton[2] - mySkeleton[0]).Unit()).Dot((otherSkeleton[2] - otherSkeleton[0]).Unit());
        bool alignedSkeletons = dotProdSkeletons > dot_prod_th_;
        LogDebug("TracksterLinkingbySkeletons")
            << "\t Outer Trackster is Good, checking for skeleton alignment " << alignedSkeletons << " dotProd "
            << dotProdSkeletons << " Threshold " << dot_prod_th_ << std::endl;
        if (alignedSkeletons)
          LogDebug("TracksterLinkingbySkeletons") << "\t\t Linked! " << std::endl;
        return alignedSkeletons;
      } else {
        // we measure the distance between the two closest nodes in the two skeletons
        LogDebug("TracksterLinkingbySkeletons")
            << "\t Outer Trackster is not aligned,  check skeletons distances " << std::endl;
        int myClosestPoint = -1;
        int otherClosestPoint = -1;
        float minDistance_z = std::numeric_limits<float>::max();
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            float dist_z = std::abs(mySkeleton[i].Z() - otherSkeleton[j].Z());
            if (dist_z < minDistance_z) {
              myClosestPoint = i;
              otherClosestPoint = j;
              minDistance_z = dist_z;
            }
          }
        }
        if (minDistance_z < min_distance_z_[isEE]) {
          LogDebug("TracksterLinkingbySkeletons")
              << "\t Trackster have distance in Z " << minDistance_z
              << "Checking if they are aligned in projective space "
              << projective_distance(mySkeleton[myClosestPoint], otherSkeleton[otherClosestPoint]) << " TH "
              << max_distance_projective_sqr_[isEE] << std::endl;
          if (projective_distance(mySkeleton[myClosestPoint], otherSkeleton[otherClosestPoint]) <
              max_distance_projective_sqr_[isEE]) {
            LogDebug("TracksterLinkingbySkeletons") << "\t\t Linked! " << std::endl;
          }
          return projective_distance(mySkeleton[myClosestPoint], otherSkeleton[otherClosestPoint]) <
                 max_distance_projective_sqr_[isEE];
        } else {
          LogDebug("TracksterLinkingbySkeletons") << "\t\t Not Linked Distance Z " << minDistance_z << std::endl;
          return false;
        }
      }
    } else {
      LogDebug("TracksterLinkingbySkeletons")
          << "\t Outer Trackster is NOT GOOD,  check projective space alignment " << areAlignedInProjectiveSpace
          << " proj_distance " << max_distance_projective_sqr_[isEE] << std::endl;
      if (areAlignedInProjectiveSpace) {
        LogDebug("TracksterLinkingbySkeletons") << "\t\t Linked! " << std::endl;
        return true;
      } else {
        LogDebug("TracksterLinkingbySkeletons")
            << "\t Not aligned in projective space, check distance between closest points in the two skeletons "
            << std::endl;
        // we measure the distance between the two closest nodes in the two skeletons
        int myClosestPoint = -1;
        int otherClosestPoint = -1;
        float minDistance_z = std::numeric_limits<float>::max();
        // we skip the innermost node of mySkeleton
        for (int i = 1; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            float dist_z = std::abs(mySkeleton[i].Z() - otherSkeleton[j].Z());
            if (dist_z < minDistance_z) {
              myClosestPoint = i;
              otherClosestPoint = j;
              minDistance_z = dist_z;
            }
          }
        }
        float d = projective_distance(mySkeleton[myClosestPoint], otherSkeleton[otherClosestPoint]);
        LogDebug("TracksterLinkingbySkeletons")
            << "\t\t Distance between closest points " << d << " TH " << 10.f << " Z Distance " << minDistance_z
            << " TH " << max_distance_projective_sqr_closest_points_[isEE] << std::endl;
        if (d < max_distance_projective_sqr_closest_points_[isEE] and
            minDistance_z < max_z_distance_closest_points_[isEE]) {
          LogDebug("TracksterLinkingbySkeletons") << "\t\t\t Linked! " << d << std::endl;
          return true;
        } else {
          LogDebug("TracksterLinkingbySkeletons") << "Distance between closest point " << d << " Distance in z "
                                                  << max_z_distance_closest_points_[isEE] << std::endl;
          bool isInCyl = isInCylinder(mySkeleton, otherSkeleton, cylinder_radius_sqr_[isEE]);
          LogDebug("TracksterLinkingbySkeletons") << "Two Points are in Cylinder  " << isInCylinder << std::endl;
          if (isInCyl) {
            LogDebug("TracksterLinkingbySkeletons") << "\t\t\t Linked! " << d << std::endl;
          }
          return isInCyl;
        }
      }
    }
  }
}

void TracksterLinkingbySkeletons::linkTracksters(
    const Inputs &input,
    std::vector<Trackster> &resultTracksters,
    std::vector<std::vector<unsigned int>> &linkedResultTracksters,
    std::vector<std::vector<unsigned int>> &linkedTracksterIdToInputTracksterId) {
  const auto &tracksters = input.tracksters;
  const auto &layerClusters = input.layerClusters;

  // sort tracksters by energy
  std::vector<unsigned int> sortedTracksters(tracksters.size());
  std::iota(sortedTracksters.begin(), sortedTracksters.end(), 0);
  std::sort(sortedTracksters.begin(), sortedTracksters.end(), [&tracksters](unsigned int i, unsigned int j) {
    return tracksters[i].raw_energy() > tracksters[j].raw_energy();
  });
  // fill tiles for trackster linking
  // tile 0 for negative eta
  // tile 1 for positive eta
  std::array<TICLLayerTile, 2> tracksterTile;
  // loop over tracksters sorted by energy and calculate skeletons
  // fill tiles for trackster linking
  std::vector<std::array<ticl::Vector, 3>> skeletons(tracksters.size());
  for (auto const t_idx : sortedTracksters) {
    const auto &trackster = tracksters[t_idx];
    skeletons[t_idx] = findSkeletonNodes(tracksters[t_idx], 0.1, 0.9, layerClusters, rhtools_);
    tracksterTile[trackster.barycenter().eta() > 0.f].fill(
        trackster.barycenter().eta(), trackster.barycenter().phi(), t_idx);
  }
  std::vector<int> maskReceivedLink(tracksters.size(), 1);
  std::vector<int> isRootTracksters(tracksters.size(), 1);

  std::vector<Node> allNodes;
  for (size_t it = 0; it < tracksters.size(); ++it) {
    allNodes.emplace_back(it);
  }

  // loop over tracksters sorted by energy and link them
  for (auto const &t_idx : sortedTracksters) {
    auto const &trackster = tracksters[t_idx];
    auto const &skeleton = skeletons[t_idx];

    auto const bary = trackster.barycenter();
    float eta_min = std::max(abs(bary.eta()) - del_, TileConstants::minEta);
    float eta_max = std::min(abs(bary.eta()) + del_, TileConstants::maxEta);
    int tileIndex = bary.eta() > 0.f;
    const auto &tiles = tracksterTile[tileIndex];
    std::array<int, 4> search_box = tiles.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del_, bary.phi() + del_);
    if (search_box[2] > search_box[3]) {
      search_box[3] += TileConstants::nPhiBins;
    }

    for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
      for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
        auto &neighbours = tiles[tiles.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
        for (auto n : neighbours) {
          if (t_idx == n)
            continue;
          if (maskReceivedLink[n] == 0 or allNodes[t_idx].isInnerNeighbour(n))
            continue;
          if (isGoodTrackster(trackster, skeleton, min_num_lcs_, min_trackster_energy_, pca_quality_th_)) {
            LogDebug("TracksterLinkingbySkeletons")
                << "Trying to Link Trackster " << t_idx << " With Trackster " << n << std::endl;
            if (areCompatible(trackster, tracksters[n], skeleton, skeletons[n])) {
              LogDebug("TracksterLinkingbySkeletons")
                  << "\t==== LINK: Trackster " << t_idx << " Linked with Trackster " << n << std::endl;
              maskReceivedLink[n] = 0;
              allNodes[t_idx].addOuterNeighbour(n);
              allNodes[n].addInnerNeighbour(t_idx);
              isRootTracksters[n] = 0;
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
    if (comp.size() == 1) {
      if (input.tracksters[comp[0]].vertices().size() <= 3) {
        continue;
      }
    }
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
}  // linkTracksters
