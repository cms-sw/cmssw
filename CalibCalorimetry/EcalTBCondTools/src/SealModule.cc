#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CalibCalorimetry/EcalTBCondTools/interface/StoreEcalCondition.h"
#include "CalibCalorimetry/EcalTBCondTools/interface/ReprocessEcalPedestals.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(StoreEcalCondition);
DEFINE_ANOTHER_FWK_MODULE(ReprocessEcalPedestals);
