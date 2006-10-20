#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTDigiToRaw.h"
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1GTDigiToRaw);
DEFINE_ANOTHER_FWK_MODULE(L1GlobalTriggerRawToDigi);
