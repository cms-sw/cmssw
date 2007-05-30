#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/RPCTrigger/interface/RPCTrigger.h"
#include "L1Trigger/RPCTrigger/interface/RPCVHDLConeMaker.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCTrigger);
DEFINE_ANOTHER_FWK_MODULE(RPCVHDLConeMaker);
