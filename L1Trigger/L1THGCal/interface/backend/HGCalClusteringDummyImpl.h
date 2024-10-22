#ifndef __L1Trigger_L1THGCal_HGCalClusteringDummyImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringDummyImpl_h__

#include <array>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalClusteringDummyImpl {
public:
  HGCalClusteringDummyImpl(const edm::ParameterSet& conf);

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }

  void clusterizeDummy(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
                       l1t::HGCalClusterBxCollection& clusters);

private:
  double calibSF_;
  std::vector<double> layerWeights_;
  bool applyLayerWeights_;
  HGCalTriggerTools triggerTools_;

  void calibratePt(l1t::HGCalCluster& cluster);
};

#endif
