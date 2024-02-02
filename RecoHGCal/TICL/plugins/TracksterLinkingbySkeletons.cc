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
  bool isRoundTrackster(std::array<ticl::Vector,3> skeleton )
  {
    return (skeleton[0].Z() == skeleton[2].Z());
  }

  bool isGoodTrackster(const Trackster& trackster, std::array<ticl::Vector,3>& skeleton )
  {
    bool isGood = false;

    if (isRoundTrackster(skeleton) or trackster.vertices().size() < 7 or trackster.raw_energy() < 5.f)
    {
      isGood = false;
    }
    else
    {
      auto const &eigenvalues = trackster.eigenvalues();
      auto const sum = std::accumulate(std::begin(eigenvalues), std::end(eigenvalues), 0.f);
      float pcaQuality = eigenvalues[0] / sum;
      if (pcaQuality > 0.9)
      {
        isGood = true;
      }
    }
    return isGood;
  }


  //distance between skeletons
  float projective_distance( const ticl::Vector& point1, const ticl::Vector& point2)
  {
    // squared projective distance 
    float r1 = std::sqrt(point1.x() * point1.x() + point1.y() * point1.y());
    float r2_at_z1 = std::sqrt(point2.x() * point2.x() + point2.y() * point2.y()) *std::abs(point1.z()) / std::abs(point2.z());
    float delta_phi = reco::deltaPhi(point1.Phi(), point2.Phi());
    float projective_distance = (r1 - r2_at_z1) * (r1 - r2_at_z1) + r2_at_z1 * r2_at_z1 * delta_phi * delta_phi;
    return projective_distance;

  }




}


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




std::array<ticl::Vector,3> TracksterLinkingbySkeletons::findSkeletonNodes(const ticl::Trackster& trackster, 
                                  float lower_percentage,
                                  float upper_percentage,
                                  const std::vector<reco::CaloCluster>& layerClusters,
                                  const hgcal::RecHitTools& rhtools) {
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
  float barycenterZ = trackster.barycenter().Z();
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
  std::array<ticl::Vector,3> skeleton;
  int minimumDistanceInLayers = 4;
  if (outerLayerId - innerLayerId < minimumDistanceInLayers) {
     skeleton = {{trackster.barycenter(), trackster.barycenter(), trackster.barycenter()}};
  }
  else
  {
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


bool TracksterLinkingbySkeletons::areCompatible(const ticl::Trackster& myTrackster, const ticl::Trackster& otherTrackster, 
                       const std::array<ticl::Vector,3>& mySkeleton, const std::array<ticl::Vector,3>& otherSkeleton) {
  if (!isGoodTrackster(myTrackster, mySkeleton))
  {
    return false;
  }
  else
  {
    float proj_distance = projective_distance(mySkeleton[1], otherSkeleton[1]);
    bool areAlignedInProjectiveSpace = proj_distance < 0.1;

    if(isGoodTrackster(otherTrackster, otherSkeleton))
    {
      // if both tracksters are good, then we can check the projective distance between the barycenters.
      // if the barycenters are aligned, then we check that the two skeletons are aligned
      if(areAlignedInProjectiveSpace)
      {
          bool alignedSkeletons = ((mySkeleton[2] - mySkeleton[0]).Unit()).Dot((otherSkeleton[2] - otherSkeleton[0]).Unit()) > 0.95;
          return true;
      }
      else
      {
        // we measure the distance between the two closest nodes in the two skeletons
        int myClosestPoint = -1;
        int otherClosestPoint = -1; 
        float minDistance_z = std::numeric_limits<float>::max();
        for(int i = 0; i < 3; i++)
        {
          for(int j = 0; j < 3; j++)
          {
            float dist_z = std::abs(mySkeleton[i].Z() - otherSkeleton[j].Z());
            if (dist_z < minDistance_z)
            {
              myClosestPoint = i;
              otherClosestPoint = j;
              minDistance_z = dist_z;
            }
          }
        }
        if(minDistance_z < 20.f)
        {
          return projective_distance(mySkeleton[myClosestPoint], otherSkeleton[otherClosestPoint]) < 0.1;
        }
        else
        {
          return false;
        }

      }
    }
    else
    {
      if (areAlignedInProjectiveSpace)
      {
        return true;
      }
      else
      {
        // we measure the distance between the two closest nodes in the two skeletons
        int myClosestPoint = -1;
        int otherClosestPoint = -1; 
        float minDistance_z = std::numeric_limits<float>::max();
        // we skip the innermost node of mySkeleton
        for(int i = 1; i < 3; i++)
        {
          for(int j = 0; j < 3; j++)
          {
            float dist_z = std::abs(mySkeleton[i].Z() - otherSkeleton[j].Z());
            if (dist_z < minDistance_z)
            {
              myClosestPoint = i;
              otherClosestPoint = j;
              minDistance_z = dist_z;
            }
          }
        }
        float d = projective_distance(mySkeleton[myClosestPoint], otherSkeleton[otherClosestPoint]);
        return d < 2 and minDistance_z < 10.f;
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
  std::array<TICLLayerTile,2> tracksterTile;
  // loop over tracksters sorted by energy and calculate skeletons
  // fill tiles for trackster linking
  std::vector<std::array<ticl::Vector,3>> skeletons(tracksters.size());
  for (auto const &t_idx : sortedTracksters) {
    const auto& trackster = tracksters[t_idx];
    skeletons[t_idx] = findSkeletonNodes(tracksters[t_idx], 0.1, 0.9, layerClusters, rhtools_);
    tracksterTile[trackster.barycenter().eta() > 0.f].fill(trackster.barycenter().eta(), trackster.barycenter().phi(), t_idx);
  
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
    const auto& tiles = tracksterTile[tileIndex];
    std::array<int, 4> search_box =
          tiles.searchBoxEtaPhi(std::abs(bary.eta()) - del_, std::abs(bary.eta()) + del_, bary.phi() - del_, bary.phi() + del_);
      if (search_box[2] > search_box[3]) {
        search_box[3] += TileConstants::nPhiBins;
      }

      for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
        for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
          auto &neighbours = tiles[tiles.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
          for (auto n : neighbours) {
            if (t_idx == n)
              continue;
            if (maskReceivedLink[n] == 0)
                continue;

            if (areCompatible(tracksters, skeletons, t_idx, n )) {
              LogDebug("TracksterLinkingbySkeletons") << "\t==== LINK: Trackster " << t_idx << " Linked with Trackster "
                                                      << n << std::endl;
              maskReceivedLink[n] = 0;
              allNodes[t_idx].addNeighbour(n);
              isRootTracksters[n] = 0;
            }
          }
        }
      }



  }







  // if (TracksterLinkingAlgoBase::algo_verbosity_ > VerbosityLevel::Advanced)
  //   LogDebug("TracksterLinkingbySkeletons") << "------- Graph Linking ------- \n";

  // auto isPointInCone = [](const Vector &origin,
  //                         const Vector &direction,
  //                         const float halfAngle,
  //                         const float maxHeight,
  //                         const Vector &testPoint,
  //                         std::string str = "") -> bool {
  //   Vector toCheck = testPoint - origin;
  //   auto projection = toCheck.Dot(direction.Unit());
  //   auto const angle = ROOT::Math::VectorUtil::Angle(direction, toCheck);
  //   LogDebug("TracksterLinkingbySkeletons")
  //       << str << "Origin " << origin << " TestPoint " << testPoint << " projection " << projection << " maxHeight "
  //       << maxHeight << " Angle " << angle << " halfAngle " << halfAngle << std::endl;
  //   if (projection < 0.f || projection > maxHeight) {
  //     return false;
  //   }

  //   return angle < halfAngle;
  // };



  // //actual trackster-trackster linking
  // auto pcaQuality = [](const Trackster &trackster) -> float {
  //   auto const &eigenvalues = trackster.eigenvalues();
  //   auto const e0 = eigenvalues[0];
  //   auto const e1 = eigenvalues[1];
  //   auto const e2 = eigenvalues[2];
  //   auto const sum = e0 + e1 + e2;
  //   auto const normalized_e0 = e0 / sum;
  //   auto const normalized_e1 = e1 / sum;
  //   auto const normalized_e2 = e2 / sum;
  //   return normalized_e0;
  // };

  // const float halfAngle0 = angle_first_cone_;
  // const float halfAngle1 = angle_second_cone_;
  // const float halfAngle2 = angle_third_cone_;
  // const float maxHeightCone = max_height_cone_;



  // for (size_t it = 0; it < tracksters.size(); ++it) {
  //   auto const &trackster = tracksters[it];
  //   isHadron(trackster);

  //   auto pcaQ = pcaQuality(trackster);
  //   LogDebug("TracksterLinkingbySkeletons")
  //       << "DEBUG Trackster " << it << " energy " << trackster.raw_energy() << " Num verties "
  //       << trackster.vertices().size() << " PCA Quality " << pcaQ << std::endl;
  //   const float pcaQTh = pcaQ_;
  //   const unsigned int pcaQLCSize = pcaQLCSize_;
  //   if (pcaQ >= pcaQTh && trackster.vertices().size() > pcaQLCSize) {
  //     auto const skeletons = returnSkeletons(trackster);
  //     LogDebug("TracksterLinkingbySkeletons")
  //         << "Trackster " << it << " energy " << trackster.raw_energy() << " Num verties "
  //         << trackster.vertices().size() << " PCA Quality " << pcaQ << " Skeletons " << skeletons[0] << std::endl;
  //     auto const &eigenVec = trackster.eigenvectors(0);
  //     auto const eigenVal = trackster.eigenvalues()[0];
  //     auto const &directionOrigin = eigenVec * eigenVal;

  //     auto const bary = trackster.barycenter();
  //     float eta_min = std::max(abs(bary.eta()) - del_, TileConstants::minEta);
  //     float eta_max = std::min(abs(bary.eta()) + del_, TileConstants::maxEta);

  //     if (bary.eta() > 0.) {
  //       std::array<int, 4> search_box =
  //           tracksterTilePos.searchBoxEtaPhi(eta_min, eta_max, bary.phi() - del_, bary.phi() + del_);
  //       if (search_box[2] > search_box[3]) {
  //         search_box[3] += TileConstants::nPhiBins;
  //       }

  //       for (int eta_i = search_box[0]; eta_i <= search_box[1]; ++eta_i) {
  //         for (int phi_i = search_box[2]; phi_i <= search_box[3]; ++phi_i) {
  //           auto &neighbours = tracksterTilePos[tracksterTilePos.globalBin(eta_i, (phi_i % TileConstants::nPhiBins))];
  //           for (auto n : neighbours) {
  //             if (maskReceivedLink[n] == 0)
  //               continue;

  //             auto const &trackster_out = tracksters[n];
  //             auto const &skeletons_out = returnSkeletons(trackster_out);
  //             auto const skeletonDist2 = (skeletons[2] - skeletons_out[0]).Mag2();
  //             auto const pcaQOuter = pcaQuality(trackster_out);
  //             //              auto const dotProd  = ((skeletons[0]-skeletons[2]).Unit()).Dot((skeletons_out[0] - skeletons_out[2]).Unit());
  //             bool isGoodPCA = (pcaQOuter >= pcaQTh) && (trackster_out.vertices().size() > pcaQLCSize);
  //             auto const maxHeightSmallCone = std::sqrt((skeletons[2] - skeletons[0]).Mag2());
  //             bool isInSmallCone = isPointInCone(
  //                 skeletons[0], directionOrigin, halfAngle0, maxHeightSmallCone, skeletons_out[0], "Small Cone");
  //             bool isInCone =
  //                 isPointInCone(skeletons[1], directionOrigin, halfAngle1, maxHeightCone, skeletons_out[0], "BigCone ");
  //             bool isInLastCone =
  //                 isPointInCone(skeletons[2], directionOrigin, halfAngle2, maxHeightCone, skeletons_out[0], "LastCone");
  //             bool dotProd =
  //                 isGoodPCA
  //                     ? ((skeletons[0] - skeletons[2]).Unit()).Dot((skeletons_out[0] - skeletons_out[2]).Unit()) >=
  //                           dotCut_
  //                     : true;

  //             LogDebug("TracksterLinkingbySkeletons")
  //                 << "\tTrying to Link Trackster " << n << " energy " << tracksters[n].raw_energy() << " LCs "
  //                 << tracksters[n].vertices().size() << " skeletons " << skeletons_out[0] << " Dist " << skeletonDist2
  //                 << " dot Prod "
  //                 << ((skeletons[0] - skeletons[2]).Unit()).Dot((skeletons_out[0] - skeletons_out[2]).Unit())
  //                 << " isGoodDotProd " << dotProd << " isPointInBigCone " << isInCone << " isPointInSmallCone "
  //                 << isInSmallCone << " isPointInLastCone " << isInLastCone << std::endl;
  //             if (isInLastCone && dotProd) {
  //               LogDebug("TracksterLinkingbySkeletons") << "\t==== LINK: Trackster " << it << " Linked with Trackster "
  //                                                       << n << " LCs " << tracksters[n].vertices().size() << std::endl;
  //               LogDebug("TracksterLinkingbySkeletons")
  //                   << "\t\tSkeleton origin " << skeletons[2] << " Skeleton out " << skeletons_out[0] << std::endl;
  //               maskReceivedLink[n] = 0;
  //               allNodes[it].addNeighbour(n);
  //               isRootTracksters[n] = 0;
  //             }
  //             if (isInCone && skeletonDist2 <= maxDistSkeletonsSq_ && dotProd) {
  //               LogDebug("TracksterLinkingbySkeletons") << "\t==== LINK: Trackster " << it << " Linked with Trackster "
  //                                                       << n << " LCs " << tracksters[n].vertices().size() << std::endl;
  //               LogDebug("TracksterLinkingbySkeletons")
  //                   << "\t\tSkeleton origin " << skeletons[2] << " Skeleton out " << skeletons_out[0] << std::endl;
  //               maskReceivedLink[n] = 0;
  //               allNodes[it].addNeighbour(n);
  //               isRootTracksters[n] = 0;
  //               continue;
  //             }
  //             if (isInSmallCone && !isGoodPCA) {
  //               maskReceivedLink[n] = 0;
  //               allNodes[it].addNeighbour(n);
  //               isRootTracksters[n] = 0;
  //               LogDebug("TracksterLinkingbySkeletons")
  //                   << "\t==== LINK: Trackster " << it << " Linked with Trackster in small cone " << n << std::endl;
  //               LogDebug("TracksterLinkingbySkeletons")
  //                   << "\t\tSkeleton origin " << skeletons[0] << " Skeleton out " << skeletons_out[0] << std::endl;
  //               continue;
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

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
