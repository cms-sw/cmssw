#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/L1TriggerExt/plugins/L1CondDBPayloadWriterExt.h"
#include "CondTools/L1TriggerExt/plugins/L1CondDBIOVWriterExt.h"
#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyDummyProdExt.h"
#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyListDummyProdExt.h"
#include "CondTools/L1TriggerExt/plugins/L1SubsystemKeysOnlineProdExt.h"
#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyOnlineProdExt.h"


using namespace l1t;

DEFINE_FWK_MODULE(L1CondDBPayloadWriterExt);
DEFINE_FWK_MODULE(L1CondDBIOVWriterExt);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyDummyProdExt);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyListDummyProdExt);
DEFINE_FWK_EVENTSETUP_MODULE(L1SubsystemKeysOnlineProdExt);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProdExt);

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondTools/L1Trigger/interface/WriterProxy.h"



// Central L1 records
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"

REGISTER_L1_WRITER(L1TriggerKeyExtRcd, L1TriggerKeyExt);

#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"

REGISTER_L1_WRITER(L1TriggerKeyListExtRcd, L1TriggerKeyListExt);

// Ext Records:

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuO2ORcd.h"

REGISTER_L1_WRITER(L1TUtmTriggerMenuO2ORcd, L1TUtmTriggerMenu);

