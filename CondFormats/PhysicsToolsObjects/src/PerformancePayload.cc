#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"

const float PerformancePayload::InvalidResult = -100.;


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(PerformancePayload);
