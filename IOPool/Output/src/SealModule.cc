#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOPool/Output/src/PoolOutputModule.h"

using edm::PoolOutputModule;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(PoolOutputModule);
