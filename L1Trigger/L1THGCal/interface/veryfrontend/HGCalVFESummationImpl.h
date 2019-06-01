#ifndef __L1Trigger_L1THGCal_HGCalVFESummationImpl_h__
#define __L1Trigger_L1THGCal_HGCalVFESummationImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include <array>
#include <vector>

class HGCalVFESummationImpl {
public:
  HGCalVFESummationImpl(const edm::ParameterSet& conf);

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }
  void triggerCellSums(const HGCalTriggerGeometryBase&,
                       const std::vector<std::pair<DetId, uint32_t> >&,
                       std::unordered_map<uint32_t, uint32_t>& payload);

private:
  std::vector<double> thickness_corrections_;
  double lsb_silicon_fC_;
  double lsb_scintillator_MIP_;
  std::vector<double> thresholds_silicon_;
  double threshold_scintillator_;
  HGCalTriggerTools triggerTools_;
};

#endif
