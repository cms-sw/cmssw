
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmulator.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTestAnalyzer.h"
#include "L1Trigger/GlobalCaloTrigger/src/FakeGctInputProducer.h"
#include "L1Trigger/GlobalCaloTrigger/src/FakeGctInputTester.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1GctEmulator);
DEFINE_ANOTHER_FWK_MODULE(L1GctTestAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(FakeGctInputProducer);
DEFINE_ANOTHER_FWK_MODULE(FakeGctInputTester);
