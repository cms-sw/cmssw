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

#include "CondCore/ESSources/interface/registration_macros.h"
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

#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosO2ORcd.h"

REGISTER_L1_WRITER(L1TGlobalPrescalesVetosO2ORcd, L1TGlobalPrescalesVetos);

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"

REGISTER_L1_WRITER(L1TMuonBarrelParamsO2ORcd, L1TMuonBarrelParams);

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsO2ORcd.h"

REGISTER_L1_WRITER(L1TMuonEndCapParamsO2ORcd, L1TMuonEndCapParams);

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestO2ORcd.h"

REGISTER_L1_WRITER(L1TMuonEndCapForestO2ORcd, L1TMuonEndCapForest);

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsO2ORcd.h"

REGISTER_L1_WRITER(L1TMuonOverlapParamsO2ORcd, L1TMuonOverlapParams);

#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsO2ORcd.h"

REGISTER_L1_WRITER(L1TMuonGlobalParamsO2ORcd, L1TMuonGlobalParams);

#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
using namespace l1t;
REGISTER_L1_WRITER(L1TCaloParamsO2ORcd, CaloParams);
