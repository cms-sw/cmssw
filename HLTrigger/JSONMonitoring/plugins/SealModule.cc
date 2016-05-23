#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTrigger/JSONMonitoring/interface/TriggerJSONMonitoring.h"
#include "HLTrigger/JSONMonitoring/interface/HLTriggerJSONMonitoring.h"
#include "HLTrigger/JSONMonitoring/interface/L1TriggerJSONMonitoring.h"

DEFINE_FWK_MODULE(HLTriggerJSONMonitoring);
DEFINE_FWK_MODULE(L1TriggerJSONMonitoring);
DEFINE_FWK_MODULE(TriggerJSONMonitoring);
