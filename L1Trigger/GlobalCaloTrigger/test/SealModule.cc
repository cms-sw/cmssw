
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/GlobalCaloTrigger/test/L1GctTest.h"
#include "L1Trigger/GlobalCaloTrigger/test/L1GctTestAnalyzer.h"
#include "L1Trigger/GlobalCaloTrigger/test/FakeGctInputProducer.h"
#include "L1Trigger/GlobalCaloTrigger/test/FakeGctInputTester.h"

DEFINE_FWK_MODULE(L1GctTest);
DEFINE_FWK_MODULE(L1GctTestAnalyzer);
DEFINE_FWK_MODULE(FakeGctInputProducer);
DEFINE_FWK_MODULE(FakeGctInputTester);
