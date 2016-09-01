#include "L1Trigger/L1THGCal/interface/HGCalTriggerLightweightGeometryBase.h"



HGCalTriggerLightweightGeometryBase::
HGCalTriggerLightweightGeometryBase(const edm::ParameterSet& conf) : 
  name_(conf.getParameter<std::string>("TriggerGeometryName")),
  ee_sd_name_(conf.getParameter<std::string>("eeSDName")),
  fh_sd_name_(conf.getParameter<std::string>("fhSDName")),
  bh_sd_name_(conf.getParameter<std::string>("bhSDName")) 
{
}

void HGCalTriggerLightweightGeometryBase::reset() 
{
}


EDM_REGISTER_PLUGINFACTORY(HGCalTriggerLightweightGeometryFactory,
			   "HGCalTriggerLightweightGeometryFactory");
