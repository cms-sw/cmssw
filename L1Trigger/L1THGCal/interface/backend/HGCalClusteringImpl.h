#ifndef __L1Trigger_L1THGCal_HGCalClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringImpl_h__

#include <array>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

inline bool distanceSorter(pair<edm::Ptr<l1t::HGCalTriggerCell>, float> i,
                           pair<edm::Ptr<l1t::HGCalTriggerCell>, float> j) {
  return (i.second < j.second);
}

class HGCalClusteringImpl {
private:
  static constexpr unsigned kNSides_ = 2;

public:
  HGCalClusteringImpl(const edm::ParameterSet& conf);

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

  /* dR-algorithms */
  bool isPertinent(const l1t::HGCalTriggerCell& tc, const l1t::HGCalCluster& clu, double distXY) const;

  void clusterizeDR(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
                    l1t::HGCalClusterBxCollection& clusters);

  /* NN-algorithms */
  void mergeClusters(l1t::HGCalCluster& main_cluster, const l1t::HGCalCluster& secondary_cluster) const;

  void NNKernel(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& reshuffledTriggerCells,
                l1t::HGCalClusterBxCollection& clusters,
                const HGCalTriggerGeometryBase& triggerGeometry);

  void clusterizeNN(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
                    l1t::HGCalClusterBxCollection& clusters,
                    const HGCalTriggerGeometryBase& triggerGeometry);

  /* FW-algorithms */
  void clusterizeDRNN(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
                      l1t::HGCalClusterBxCollection& clusters,
                      const HGCalTriggerGeometryBase& triggerGeometry);

private:
  double siliconSeedThreshold_;
  double siliconTriggerCellThreshold_;
  double scintillatorSeedThreshold_;
  double scintillatorTriggerCellThreshold_;
  double dr_;
  std::string clusteringAlgorithmType_;
  double calibSF_;
  std::vector<double> layerWeights_;
  bool applyLayerWeights_;
  HGCalTriggerTools triggerTools_;

  void triggerCellReshuffling(
      const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
      std::array<std::vector<std::vector<edm::Ptr<l1t::HGCalTriggerCell>>>, kNSides_>& reshuffledTriggerCells);

  bool areTCneighbour(uint32_t detIDa, uint32_t detIDb, const HGCalTriggerGeometryBase& triggerGeometry);

  void removeUnconnectedTCinCluster(l1t::HGCalCluster& cluster, const HGCalTriggerGeometryBase& triggerGeometry);

  void calibratePt(l1t::HGCalCluster& cluster);
};

#endif
