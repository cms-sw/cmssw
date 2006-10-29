#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/DTTrigger/test/DTTrigTest.cc"
#include "L1Trigger/DTTrigger/src/DTTrigProd.cc"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTTrigTest);
DEFINE_ANOTHER_FWK_MODULE(DTTrigProd);
