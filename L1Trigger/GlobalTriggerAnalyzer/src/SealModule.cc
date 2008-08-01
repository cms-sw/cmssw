#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtAnalyzer.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReport.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPackUnpackAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1GtAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L1GtTrigReport);
DEFINE_ANOTHER_FWK_MODULE(L1GtPackUnpackAnalyzer);
