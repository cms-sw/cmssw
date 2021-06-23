#ifndef __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
#define __RecoParticleFlow_PFClusterProducer_RealisticHitToClusterAssociator_H__
/////////////////////////
// Author: Felice Pantaleo
// Date:   30/06/2017
// Email: felice@cern.ch
/////////////////////////
#include <vector>
#include <cmath>
#include <unordered_map>
#include "RealisticCluster.h"

class RealisticHitToClusterAssociator {
  using Hit3DPosition = std::array<float, 3>;

public:
  struct RealisticHit {
    struct HitToCluster {
      unsigned int simClusterId_;
      float mcEnergyFraction_;
      float distanceFromMaxHit_;
      float realisticEnergyFraction_;
    };

    Hit3DPosition hitPosition_;
    float totalEnergy_;
    unsigned int layerId_;
    std::vector<HitToCluster> hitToCluster_;
  };

  void init(std::size_t numberOfHits, std::size_t numberOfSimClusters, std::size_t numberOfLayers) {
    realisticHits_.resize(numberOfHits);
    realisticSimClusters_.resize(numberOfSimClusters);
    for (auto& sc : realisticSimClusters_)
      sc.setLayersNum(numberOfLayers);
  }

  void insertHitPosition(float x, float y, float z, unsigned int hitIndex) {
    realisticHits_[hitIndex].hitPosition_ = {{x, y, z}};
  }

  void insertLayerId(unsigned int layerId, unsigned int hitIndex) { realisticHits_[hitIndex].layerId_ = layerId; }

  void insertHitEnergy(float energy, unsigned int hitIndex) { realisticHits_[hitIndex].totalEnergy_ = energy; }

  void insertSimClusterIdAndFraction(unsigned int scIdx, float fraction, unsigned int hitIndex, float associatedEnergy) {
    realisticHits_[hitIndex].hitToCluster_.emplace_back(RealisticHit::HitToCluster{scIdx, fraction, 0.f, 0.f});
    realisticSimClusters_[scIdx].setMaxEnergyHit(
        realisticHits_[hitIndex].layerId_, associatedEnergy, realisticHits_[hitIndex].hitPosition_);
  }

  float XYdistanceFromMaxHit(unsigned int hitId, unsigned int clusterId) {
    auto l = realisticHits_[hitId].layerId_;
    const auto& maxHitPosition = realisticSimClusters_[clusterId].getMaxEnergyPosition(l);
    float distanceSquared = std::pow((realisticHits_[hitId].hitPosition_[0] - maxHitPosition[0]), 2) +
                            std::pow((realisticHits_[hitId].hitPosition_[1] - maxHitPosition[1]), 2);
    return std::sqrt(distanceSquared);
  }

  float XYdistanceFromPointOnSameLayer(unsigned int hitId, const Hit3DPosition& point) {
    float distanceSquared = std::pow((realisticHits_[hitId].hitPosition_[0] - point[0]), 2) +
                            std::pow((realisticHits_[hitId].hitPosition_[1] - point[1]), 2);
    return std::sqrt(distanceSquared);
  }

  void computeAssociation(float exclusiveFraction,
                          bool useMCFractionsForExclEnergy,
                          unsigned int fhOffset,
                          unsigned int bhOffset) {
    //if more than exclusiveFraction of a hit's energy belongs to a cluster, that rechit is not counted as shared
    unsigned int numberOfHits = realisticHits_.size();
    std::vector<float> partialEnergies;
    for (unsigned int hitId = 0; hitId < numberOfHits; ++hitId) {
      partialEnergies.clear();
      std::vector<unsigned int> removeAssociation;
      auto& realisticHit = realisticHits_[hitId];
      unsigned int numberOfClusters = realisticHit.hitToCluster_.size();
      if (numberOfClusters == 1) {
        unsigned int simClusterId = realisticHit.hitToCluster_[0].simClusterId_;
        float assignedFraction = 1.f;
        realisticHit.hitToCluster_[0].realisticEnergyFraction_ = assignedFraction;
        float assignedEnergy = realisticHit.totalEnergy_;
        realisticSimClusters_[simClusterId].increaseEnergy(assignedEnergy);
        realisticSimClusters_[simClusterId].addHitAndFraction(hitId, assignedFraction);
        realisticSimClusters_[simClusterId].increaseExclusiveEnergy(assignedEnergy);
      } else {
        partialEnergies.resize(numberOfClusters, 0.f);
        unsigned int layer = realisticHit.layerId_;
        float sumE = 0.f;
        float energyDecayLength = getDecayLength(layer, fhOffset, bhOffset);
        for (unsigned int clId = 0; clId < numberOfClusters; ++clId) {
          auto simClusterId = realisticHit.hitToCluster_[clId].simClusterId_;
          realisticHit.hitToCluster_[clId].distanceFromMaxHit_ = XYdistanceFromMaxHit(hitId, simClusterId);
          // partial energy is computed based on the distance from the maximum energy hit and its energy
          // partial energy is only needed to compute a fraction and it's not the energy assigned to the cluster
          auto maxEnergyOnLayer = realisticSimClusters_[simClusterId].getMaxEnergy(layer);
          if (maxEnergyOnLayer > 0.f) {
            partialEnergies[clId] =
                maxEnergyOnLayer * std::exp(-realisticHit.hitToCluster_[clId].distanceFromMaxHit_ / energyDecayLength);
          }
          sumE += partialEnergies[clId];
        }
        if (sumE > 0.f) {
          float invSumE = 1.f / sumE;
          for (unsigned int clId = 0; clId < numberOfClusters; ++clId) {
            unsigned int simClusterIndex = realisticHit.hitToCluster_[clId].simClusterId_;
            float assignedFraction = partialEnergies[clId] * invSumE;
            if (assignedFraction > 1e-3) {
              realisticHit.hitToCluster_[clId].realisticEnergyFraction_ = assignedFraction;
              float assignedEnergy = assignedFraction * realisticHit.totalEnergy_;
              realisticSimClusters_[simClusterIndex].increaseEnergy(assignedEnergy);
              realisticSimClusters_[simClusterIndex].addHitAndFraction(hitId, assignedFraction);
              // if the hits energy belongs for more than exclusiveFraction to a cluster, also the cluster's
              // exclusive energy is increased. The exclusive energy will be needed to evaluate if
              // a realistic cluster will be invisible, i.e. absorbed by other clusters
              if ((useMCFractionsForExclEnergy and
                   realisticHit.hitToCluster_[clId].mcEnergyFraction_ > exclusiveFraction) or
                  (!useMCFractionsForExclEnergy and assignedFraction > exclusiveFraction)) {
                realisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(assignedEnergy);
              }
            } else {
              removeAssociation.push_back(simClusterIndex);
            }
          }
        }
      }

      while (!removeAssociation.empty()) {
        auto clusterToRemove = removeAssociation.back();
        removeAssociation.pop_back();

        realisticHit.hitToCluster_.erase(std::remove_if(realisticHit.hitToCluster_.begin(),
                                                        realisticHit.hitToCluster_.end(),
                                                        [clusterToRemove](const RealisticHit::HitToCluster& x) {
                                                          return x.simClusterId_ == clusterToRemove;
                                                        }),
                                         realisticHit.hitToCluster_.end());
      }
    }
  }

  void findAndMergeInvisibleClusters(float invisibleFraction, float exclusiveFraction) {
    unsigned int numberOfRealSimClusters = realisticSimClusters_.size();

    for (unsigned int clId = 0; clId < numberOfRealSimClusters; ++clId) {
      if (realisticSimClusters_[clId].getExclusiveEnergyFraction() < invisibleFraction) {
        realisticSimClusters_[clId].setVisible(false);
        auto& hAndF = realisticSimClusters_[clId].hitsIdsAndFractions();
        std::unordered_map<unsigned int, float> energyInNeighbors;
        float totalSharedEnergy = 0.f;

        for (auto& elt : hAndF) {
          unsigned int hitId = elt.first;
          float fraction = elt.second;
          auto& realisticHit = realisticHits_[hitId];

          if (realisticHit.hitToCluster_.size() > 1 && fraction < 1.f) {
            float correction = 1.f - fraction;
            unsigned int numberOfClusters = realisticHit.hitToCluster_.size();
            int clusterToRemove = -1;
            for (unsigned int i = 0; i < numberOfClusters; ++i) {
              auto simClusterIndex = realisticHit.hitToCluster_[i].simClusterId_;
              if (simClusterIndex == clId) {
                clusterToRemove = i;
              } else if (realisticSimClusters_[simClusterIndex].isVisible()) {
                float oldFraction = realisticHit.hitToCluster_[i].realisticEnergyFraction_;
                float newFraction = oldFraction / correction;
                float oldEnergy = oldFraction * realisticHit.totalEnergy_;
                float newEnergy = newFraction * realisticHit.totalEnergy_;
                float sharedEnergy = newEnergy - oldEnergy;
                energyInNeighbors[simClusterIndex] += sharedEnergy;
                totalSharedEnergy += sharedEnergy;
                realisticSimClusters_[simClusterIndex].increaseEnergy(sharedEnergy);
                realisticSimClusters_[simClusterIndex].modifyFractionForHitId(newFraction, hitId);
                realisticHit.hitToCluster_[i].realisticEnergyFraction_ = newFraction;
                if (newFraction > exclusiveFraction) {
                  realisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(sharedEnergy);
                  if (oldFraction <= exclusiveFraction) {
                    realisticSimClusters_[simClusterIndex].increaseExclusiveEnergy(oldEnergy);
                  }
                }
              }
            }
            realisticSimClusters_[realisticHit.hitToCluster_[clusterToRemove].simClusterId_].modifyFractionForHitId(
                0.f, hitId);
            realisticHit.hitToCluster_.erase(realisticHit.hitToCluster_.begin() + clusterToRemove);
          }
        }

        for (auto& elt : hAndF) {
          unsigned int hitId = elt.first;
          auto& realisticHit = realisticHits_[hitId];
          if (realisticHit.hitToCluster_.size() == 1 and realisticHit.hitToCluster_[0].simClusterId_ == clId and
              totalSharedEnergy > 0.f) {
            for (auto& pair : energyInNeighbors) {
              // hits that belonged completely to the absorbed cluster are redistributed
              // based on the fraction of energy shared in the shared hits
              float sharedFraction = pair.second / totalSharedEnergy;
              if (sharedFraction > 1e-6) {
                float assignedEnergy = realisticHit.totalEnergy_ * sharedFraction;
                realisticSimClusters_[pair.first].increaseEnergy(assignedEnergy);
                realisticSimClusters_[pair.first].addHitAndFraction(hitId, sharedFraction);
                realisticHit.hitToCluster_.emplace_back(
                    RealisticHit::HitToCluster{pair.first, 0.f, -1.f, sharedFraction});
                if (sharedFraction > exclusiveFraction)
                  realisticSimClusters_[pair.first].increaseExclusiveEnergy(assignedEnergy);
              }
            }
          }
        }
      }
    }
  }

  void findCentersOfGravity() {
    for (auto& cluster : realisticSimClusters_) {
      if (cluster.isVisible()) {
        unsigned int layersNum = cluster.getLayersNum();
        std::vector<float> totalEnergyPerLayer(layersNum, 0.f);
        std::vector<float> xEnergyPerLayer(layersNum, 0.f);
        std::vector<float> yEnergyPerLayer(layersNum, 0.f);
        std::vector<float> zPositionPerLayer(layersNum, 0.f);
        const auto& hAndF = cluster.hitsIdsAndFractions();
        for (auto& elt : hAndF) {
          auto hitId = elt.first;
          auto fraction = elt.second;
          const auto& hit = realisticHits_[hitId];
          const auto& hitPos = hit.hitPosition_;
          auto layerId = hit.layerId_;
          auto hitEinCluster = hit.totalEnergy_ * fraction;
          totalEnergyPerLayer[layerId] += hitEinCluster;
          xEnergyPerLayer[layerId] += hitPos[0] * hitEinCluster;
          yEnergyPerLayer[layerId] += hitPos[1] * hitEinCluster;
          zPositionPerLayer[layerId] = hitPos[2];
        }
        Hit3DPosition centerOfGravity;
        for (unsigned int layerId = 0; layerId < layersNum; layerId++) {
          auto energyOnLayer = totalEnergyPerLayer[layerId];
          if (energyOnLayer > 0.f) {
            centerOfGravity = {{xEnergyPerLayer[layerId] / energyOnLayer,
                                yEnergyPerLayer[layerId] / energyOnLayer,
                                zPositionPerLayer[layerId]}};
            cluster.setCenterOfGravity(layerId, centerOfGravity);
          }
        }
      }
    }
  }

  void filterHitsByDistance(float maxDistance) {
    for (auto& cluster : realisticSimClusters_) {
      if (cluster.isVisible()) {
        auto& hAndF = cluster.hitsIdsAndFractions();
        for (unsigned int i = 0; i < hAndF.size(); ++i) {
          auto hitId = hAndF[i].first;
          const auto& hit = realisticHits_[hitId];
          auto layerId = hit.layerId_;
          if (XYdistanceFromPointOnSameLayer(hitId, cluster.getCenterOfGravity(layerId)) > maxDistance) {
            cluster.increaseEnergy(-hit.totalEnergy_ * hAndF[i].second);
            cluster.modifyFractionByIndex(0.f, i);
          }
        }
      }
    }
  }

  const std::vector<RealisticCluster>& realisticClusters() const { return realisticSimClusters_; }

private:
  static float getDecayLength(unsigned int layer, unsigned int fhOffset, unsigned int bhOffset) {
    constexpr float eeDecayLengthInLayer = 2.f;
    constexpr float fhDecayLengthInLayer = 1.5f;
    constexpr float bhDecayLengthInLayer = 1.f;

    if (layer <= fhOffset)
      return eeDecayLengthInLayer;
    else if (layer > fhOffset && layer <= bhOffset)
      return fhDecayLengthInLayer;
    else
      return bhDecayLengthInLayer;
  }

  // the vector of the Realistic SimClusters
  std::vector<RealisticCluster> realisticSimClusters_;
  // the vector of the Realistic SimClusters
  std::vector<RealisticHit> realisticHits_;
};

#endif
