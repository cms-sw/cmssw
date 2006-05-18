
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProducer.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTestAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1GctProducer)
DEFINE_ANOTHER_FWK_MODULE(L1GctTestAnalyzer)
