#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SecondaryInputSourceMacros.h"
#include "IOPool/SecondaryInput/src/PoolSecondarySource.h"

using edm::PoolSecondarySource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SECONDARY_INPUT_SOURCE(PoolSecondarySource)
