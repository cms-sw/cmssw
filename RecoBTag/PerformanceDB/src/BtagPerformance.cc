#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"

float BtagPerformance::getResult(BtagResult::BtagResultType r, BtagBinningPointByMap p) const {
  return pl.getResult(r,p);
}

bool BtagPerformance::isResultOk(BtagResult::BtagResultType r, BtagBinningPointByMap p) const {
  return pl.isInPayload(r,p);
}


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(BtagPerformance);
