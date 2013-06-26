#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReport.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPackUnpackAnalyzer.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtDataEmulAnalyzer.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtAnalyzer.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternGenerator.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtBeamModeFilter.h"

DEFINE_FWK_MODULE(L1GtTrigReport);
DEFINE_FWK_MODULE(L1GtPackUnpackAnalyzer);
DEFINE_FWK_MODULE(L1GtDataEmulAnalyzer);
DEFINE_FWK_MODULE(L1GtAnalyzer);
DEFINE_FWK_MODULE(L1GtPatternGenerator);
DEFINE_FWK_MODULE(L1GtBeamModeFilter);
