#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"



HGCalTriggerGeometryBase::
HGCalTriggerGeometryBase(const edm::ParameterSet& conf) : 
  name_(conf.getParameter<std::string>("TriggerGeometryName")),
  ee_sd_name_(conf.getParameter<std::string>("eeSDName")),
  fh_sd_name_(conf.getParameter<std::string>("fhSDName")),
  bh_sd_name_(conf.getParameter<std::string>("bhSDName")) 
{
}

void HGCalTriggerGeometryBase::reset() 
{
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(HGCalTriggerGeometryBase);

EDM_REGISTER_PLUGINFACTORY(HGCalTriggerGeometryFactory,
			   "HGCalTriggerGeometryFactory");
