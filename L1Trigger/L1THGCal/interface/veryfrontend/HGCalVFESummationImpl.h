#ifndef __L1Trigger_L1THGCal_HGCalVFESummationImpl_h__
#define __L1Trigger_L1THGCal_HGCalVFESummationImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include <vector>
#include <utility>
#include <unordered_map>

class HGCalVFESummationImpl {
public:
  HGCalVFESummationImpl(const edm::ParameterSet& conf);

  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }
  void triggerCellSums(const std::vector<std::pair<DetId, uint32_t> >&, std::unordered_map<uint32_t, uint32_t>&);

private:
  double lsb_silicon_fC_;
  double lsb_scintillator_MIP_;
  std::vector<double> thresholds_silicon_;
  double threshold_scintillator_;

  HGCalTriggerTools triggerTools_;
};

#endif
