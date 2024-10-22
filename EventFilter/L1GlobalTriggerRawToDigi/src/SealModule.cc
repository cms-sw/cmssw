#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTDigiToRaw.h"
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GtTextToRaw.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTEvmDigiToRaw.h"
#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerEvmRawToDigi.h"

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRecordProducer.h"

DEFINE_FWK_MODULE(L1GTDigiToRaw);
DEFINE_FWK_MODULE(L1GlobalTriggerRawToDigi);

DEFINE_FWK_MODULE(L1GtTextToRaw);

DEFINE_FWK_MODULE(L1GTEvmDigiToRaw);
DEFINE_FWK_MODULE(L1GlobalTriggerEvmRawToDigi);
DEFINE_FWK_MODULE(L1GlobalTriggerRecordProducer);

#include "EventFilter/L1GlobalTriggerRawToDigi/interface/ConditionDumperInEdm.h"

DEFINE_FWK_MODULE(ConditionDumperInEdm);
