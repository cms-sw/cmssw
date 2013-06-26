//#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "L1Trigger/RPCTrigger/interface/RPCConeBuilder.h"
#include "L1Trigger/RPCTrigger/interface/RPCTrigger.h"


DEFINE_FWK_EVENTSETUP_MODULE(RPCConeBuilder);
DEFINE_FWK_MODULE(RPCTrigger);
