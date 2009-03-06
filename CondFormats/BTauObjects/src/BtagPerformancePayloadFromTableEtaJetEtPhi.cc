#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTableEtaJetEtPhi.h"
#include "FWCore/Utilities/interface/Exception.h"

BtagPerformancePayloadFromTableEtaJetEtPhi::BtagPerformancePayloadFromTableEtaJetEtPhi(int s, std::string c,std::vector<float> t): BtagPerformancePayloadFromTable(s, c, t) {
  if (s != 12) {
    throw cms::Exception("Trying to construct a BtagPerformancePayloadFromTableEtaJetEtPhi from a wrong payload");
  }
}
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
EVENTSETUP_DATA_REG(BtagPerformancePayloadFromTableEtaJetEtPhi);
