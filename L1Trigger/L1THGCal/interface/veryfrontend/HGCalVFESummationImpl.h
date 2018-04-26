#ifndef __L1Trigger_L1THGCal_HGCalVFESummationImpl_h__
#define __L1Trigger_L1THGCal_HGCalVFESummationImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"

#include <array>
#include <vector>

class HGCalVFESummationImpl
{

 public:
  HGCalVFESummationImpl(const edm::ParameterSet& conf);
  
  void triggerCellSums(const HGCalTriggerGeometryBase& ,
                       const std::vector<std::pair<DetId, uint32_t > >&,
                       std::map<HGCalDetId, uint32_t>& payload);
  const std::vector<double>& ThicknessCorrections() const {return thickness_corrections_;} 
   
 private:
  std::vector<double> thickness_corrections_;   
};

#endif
