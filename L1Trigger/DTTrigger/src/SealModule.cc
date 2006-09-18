#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/DTTrigger/interface/DTTrigTest.h"
#include "L1Trigger/DTTrigger/interface/DTTrigProd.h"
#include "L1Trigger/DTTrigger/interface/DTTrigFineSync.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTTrigTest);
DEFINE_ANOTHER_FWK_MODULE(DTTrigFineSync);
DEFINE_ANOTHER_FWK_MODULE(DTTrigProd);
