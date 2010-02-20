#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"

const float PerformancePayload::InvalidResult = -100.;


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(PerformancePayload);
