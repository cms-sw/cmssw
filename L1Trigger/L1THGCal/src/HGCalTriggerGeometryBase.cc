#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

HGCalTriggerGeometryBase::HGCalTriggerGeometryBase(const edm::ParameterSet& conf)
    : name_(conf.getParameter<std::string>("TriggerGeometryName")) {}

void HGCalTriggerGeometryBase::reset() {}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(HGCalTriggerGeometryBase);

EDM_REGISTER_PLUGINFACTORY(HGCalTriggerGeometryFactory, "HGCalTriggerGeometryFactory");
