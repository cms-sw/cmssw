#ifndef __L1Trigger_L1THGCal_HGCalHistoClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalHistoClusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"

class HGCalHistoClusteringImpl {
public:
  HGCalHistoClusteringImpl(const edm::ParameterSet& conf);

  void setGeometry(const HGCalTriggerGeometryBase* const geom) {
    triggerTools_.setGeometry(geom);
    shape_.setGeometry(geom);
    if ((!dr_byLayer_coefficientA_.empty() && (dr_byLayer_coefficientA_.size() - 1) < triggerTools_.lastLayerBH()) ||
        (!dr_byLayer_coefficientB_.empty() && (dr_byLayer_coefficientB_.size() - 1) < triggerTools_.lastLayerBH())) {
      throw cms::Exception("Configuration")
          << "The per-layer dR values go up to " << (dr_byLayer_coefficientA_.size() - 1) << "(A) and "
          << (dr_byLayer_coefficientB_.size() - 1) << "(B), while layers go up to " << triggerTools_.lastLayerBH()
          << "\n";
    }
  }

  float dR(const l1t::HGCalCluster& clu, const GlobalPoint& seed) const;

  void clusterizeHisto(const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtr,
                       const std::vector<std::pair<GlobalPoint, double>>& seedPositionsEnergy,
                       const HGCalTriggerGeometryBase& triggerGeometry,
                       l1t::HGCalMulticlusterBxCollection& multiclusters,
                       l1t::HGCalClusterBxCollection& rejected_clusters) const;

private:
  enum ClusterAssociationStrategy { NearestNeighbour, EnergySplit };

  std::vector<l1t::HGCalMulticluster> clusterSeedMulticluster(
      const std::vector<edm::Ptr<l1t::HGCalCluster>>& clustersPtrs,
      const std::vector<std::pair<GlobalPoint, double>>& seeds,
      std::vector<l1t::HGCalCluster>& rejected_clusters) const;

  void finalizeClusters(std::vector<l1t::HGCalMulticluster>&,
                        const std::vector<l1t::HGCalCluster>&,
                        l1t::HGCalMulticlusterBxCollection&,
                        l1t::HGCalClusterBxCollection&,
                        const HGCalTriggerGeometryBase&) const;

  double dr_;
  std::vector<double> dr_byLayer_coefficientA_;
  std::vector<double> dr_byLayer_coefficientB_;
  double ptC3dThreshold_;

  std::string cluster_association_input_;
  ClusterAssociationStrategy cluster_association_strategy_;

  HGCalShowerShape shape_;
  HGCalTriggerTools triggerTools_;
  std::unique_ptr<HGCalTriggerClusterIdentificationBase> id_;

  static constexpr double kMidRadius_ = 2.3;
};

#endif
