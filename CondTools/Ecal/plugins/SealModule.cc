#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/Ecal/plugins/StoreEcalCondition.h"
#include "CondTools/Ecal/plugins/StoreESCondition.h"
#include "CondTools/Ecal/interface/EcalDBCopy.h"
#include "CondTools/Ecal/interface/ESDBCopy.h"
#include "CondTools/Ecal/interface/EcalTestDevDB.h"
#include "CondTools/Ecal/interface/EcalGetLaserData.h"

#include "CondCore/PopCon/interface/PopConAnalyzer.h"


DEFINE_FWK_MODULE(StoreEcalCondition);
DEFINE_FWK_MODULE(StoreESCondition);
DEFINE_FWK_MODULE(EcalDBCopy);
DEFINE_FWK_MODULE(ESDBCopy);
DEFINE_FWK_MODULE(EcalTestDevDB);
DEFINE_FWK_MODULE(EcalGetLaserData);

