#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/Ecal/interface/StoreEcalCondition.h"
#include "CondTools/Ecal/interface/ReprocessEcalPedestals.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(StoreEcalCondition);
DEFINE_ANOTHER_FWK_MODULE(ReprocessEcalPedestals);
