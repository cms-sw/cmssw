#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTestProducer.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTestAnalyzer.h"
#include "L1Trigger/RegionalCaloTrigger/src/checkTPGs.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1RCTProducer);
DEFINE_ANOTHER_FWK_MODULE(L1RCTTestAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L1RCTTestProducer);
DEFINE_ANOTHER_FWK_MODULE(checkTPGs);
