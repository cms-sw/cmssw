#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "IOPool/Input/src/PoolSource.h"

using edm::PoolRASource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(PoolRASource)
