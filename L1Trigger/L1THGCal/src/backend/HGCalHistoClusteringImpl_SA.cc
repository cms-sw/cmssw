#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoClusteringImpl_SA.h"

#include <map>
#include <cmath>

std::vector<l1thgcfirmware::HGCalMulticluster> HGCalHistoClusteringImplSA::clusterSeedMulticluster_SA(
    const std::vector<l1thgcfirmware::HGCalCluster>& clusters,
    const std::vector<l1thgcfirmware::HGCalSeed>& seeds,
    std::vector<l1thgcfirmware::HGCalCluster>& rejected_clusters,
    const l1thgcfirmware::ClusterAlgoConfig& configuration) const {
  std::map<int, l1thgcfirmware::HGCalMulticluster> mapSeedMulticluster;
  std::vector<l1thgcfirmware::HGCalMulticluster> multiclustersOut;

  for (const auto& clu : clusters) {
    int z_side = clu.zside();

    double radiusCoefficientA = configuration.dr_byLayer_coefficientA().empty()
                                    ? configuration.dr()
                                    : configuration.dr_byLayer_coefficientA()[clu.layer()];
    double radiusCoefficientB =
        configuration.dr_byLayer_coefficientB().empty() ? 0 : configuration.dr_byLayer_coefficientB()[clu.layer()];

    double minDistSqrd = radiusCoefficientA + radiusCoefficientB * (configuration.midRadius() - std::abs(clu.eta()));
    minDistSqrd *= minDistSqrd;

    std::vector<std::pair<int, double>> targetSeedsEnergy;

    unsigned int iseed = 0;
    for (const auto& seed : seeds) {
      if (z_side * seed.z() < 0) {
        ++iseed;
        continue;
      }

      double seedEnergy = seed.energy();

      double d = (clu.x() - seed.x()) * (clu.x() - seed.x()) + (clu.y() - seed.y()) * (clu.y() - seed.y());

      if (d < minDistSqrd) {
        // NearestNeighbour
        minDistSqrd = d;

        if (targetSeedsEnergy.empty()) {
          targetSeedsEnergy.emplace_back(iseed, seedEnergy);
        } else {
          targetSeedsEnergy.at(0).first = iseed;
          targetSeedsEnergy.at(0).second = seedEnergy;
        }
      }
      ++iseed;
    }

    if (targetSeedsEnergy.empty()) {
      rejected_clusters.emplace_back(clu);
      continue;
    }

    // N.B. as I have only implemented NearestNeighbour option
    // then targetSeedsEnergy has at most 1 seed for this cluster
    // Leaving in some redundant functionality in case we need
    // EnergySplit option

    for (const auto& energy : targetSeedsEnergy) {
      double seedWeight = 1;
      if (mapSeedMulticluster[energy.first].size() == 0) {
        mapSeedMulticluster[energy.first] = l1thgcfirmware::HGCalMulticluster(clu, 1);
      } else {
        mapSeedMulticluster[energy.first].addConstituent(clu, true, seedWeight);
      }
    }
  }

  multiclustersOut.reserve(mapSeedMulticluster.size());
  for (const auto& mclu : mapSeedMulticluster)
    multiclustersOut.emplace_back(mclu.second);

  return multiclustersOut;
}

void HGCalHistoClusteringImplSA::finalizeClusters_SA(
    const std::vector<l1thgcfirmware::HGCalMulticluster>& multiclusters_in,
    const std::vector<l1thgcfirmware::HGCalCluster>& rejected_clusters_in,
    std::vector<l1thgcfirmware::HGCalMulticluster>& multiclusters_out,
    std::vector<l1thgcfirmware::HGCalCluster>& rejected_clusters_out,
    const l1thgcfirmware::ClusterAlgoConfig& configuration) const {
  for (const auto& tc : rejected_clusters_in) {
    rejected_clusters_out.push_back(tc);
  }

  for (const auto& multicluster : multiclusters_in) {
    if (multicluster.sumPt() > configuration.ptC3dThreshold()) {
      multiclusters_out.push_back(multicluster);
    } else {
      for (const auto& tc : multicluster.constituents()) {
        rejected_clusters_out.push_back(tc);
      }
    }
  }
}
