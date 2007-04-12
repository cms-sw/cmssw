#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/VectorInputSourceMacros.h"
#include "IOPool/Input/src/PoolSource.h"

using edm::PoolSource;
using edm::PoolRASource;
DEFINE_FWK_INPUT_SOURCE(PoolSource);
DEFINE_FWK_VECTOR_INPUT_SOURCE(PoolSource);
DEFINE_FWK_INPUT_SOURCE(PoolRASource);
DEFINE_FWK_VECTOR_INPUT_SOURCE(PoolRASource);
