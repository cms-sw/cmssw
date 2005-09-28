#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "IOPool/SecondaryInput/src/PoolSecondarySource.h"

using edm::PoolSecondarySource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(PoolSecondarySource)
