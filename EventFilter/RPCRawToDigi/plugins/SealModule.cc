#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RPCUnpackingModule.h"
#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCUnpackingModule);
DEFINE_ANOTHER_FWK_MODULE(RPCPackingModule);
