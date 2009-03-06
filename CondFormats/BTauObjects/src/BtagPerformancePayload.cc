#include "CondFormats/BTauObjects/interface/BtagPerformancePayload.h"

const float BtagPerformancePayload::InvalidResult = -100.;


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(BtagPerformancePayload);
