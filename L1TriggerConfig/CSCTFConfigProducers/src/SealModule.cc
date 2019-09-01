#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include <L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFConfigProducer.h>
#include <L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFObjectKeysOnlineProd.h>
#include <L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFConfigOnlineProd.h>
#include <L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFAlignmentOnlineProd.h>
#include <L1TriggerConfig/CSCTFConfigProducers/interface/L1MuCSCTFParametersTester.h>

DEFINE_FWK_EVENTSETUP_MODULE(CSCTFConfigProducer);
DEFINE_FWK_EVENTSETUP_MODULE(CSCTFObjectKeysOnlineProd);
DEFINE_FWK_EVENTSETUP_MODULE(CSCTFConfigOnlineProd);
DEFINE_FWK_EVENTSETUP_MODULE(CSCTFAlignmentOnlineProd);
DEFINE_FWK_MODULE(L1MuCSCTFParametersTester);
