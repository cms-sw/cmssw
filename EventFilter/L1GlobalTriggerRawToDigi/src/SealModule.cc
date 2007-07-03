#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTDigiToRaw.h"
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GtTextToRaw.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTEvmDigiToRaw.h"
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerEvmRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1GTDigiToRaw);
DEFINE_ANOTHER_FWK_MODULE(L1GlobalTriggerRawToDigi);

DEFINE_ANOTHER_FWK_MODULE(L1GtTextToRaw);

DEFINE_ANOTHER_FWK_MODULE(L1GTEvmDigiToRaw);
DEFINE_ANOTHER_FWK_MODULE(L1GlobalTriggerEvmRawToDigi);
