#ifndef __L1Trigger_L1THGCal_HGCalHistoClusteringImplSA_h__
#define __L1Trigger_L1THGCal_HGCalHistoClusteringImplSA_h__

#include "L1Trigger/L1THGCal/interface/backend/HGCalCluster_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalSeed_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticluster_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalHistoClusteringConfig_SA.h"

#include <string>
#include <vector>
#include <memory>

class HGCalHistoClusteringImplSA {
public:
  HGCalHistoClusteringImplSA() = default;
  ~HGCalHistoClusteringImplSA() = default;

  void runAlgorithm() const;

  std::vector<l1thgcfirmware::HGCalMulticluster> clusterSeedMulticluster_SA(
      const std::vector<l1thgcfirmware::HGCalCluster>& clusters,
      const std::vector<l1thgcfirmware::HGCalSeed>& seeds,
      std::vector<l1thgcfirmware::HGCalCluster>& rejected_clusters,
      const l1thgcfirmware::ClusterAlgoConfig& configuration) const;

  void finalizeClusters_SA(const std::vector<l1thgcfirmware::HGCalMulticluster>&,
                           const std::vector<l1thgcfirmware::HGCalCluster>&,
                           std::vector<l1thgcfirmware::HGCalMulticluster>&,
                           std::vector<l1thgcfirmware::HGCalCluster>&,
                           const l1thgcfirmware::ClusterAlgoConfig& configuration) const;
};

#endif
