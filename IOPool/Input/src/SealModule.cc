#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/VectorInputSourceMacros.h"
#include "IOPool/Input/src/PoolSource.h"

using edm::PoolSource;
using edm::PoolRASource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(PoolSource);
DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(PoolSource);
DEFINE_ANOTHER_FWK_INPUT_SOURCE(PoolRASource);
DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(PoolRASource);
