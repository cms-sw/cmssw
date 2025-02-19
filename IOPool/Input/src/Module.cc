#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Sources/interface/VectorInputSourceMacros.h"
#include "PoolSource.h"

using edm::PoolSource;
using edm::PoolRASource;
DEFINE_FWK_INPUT_SOURCE(PoolSource);
DEFINE_FWK_VECTOR_INPUT_SOURCE(PoolSource);
DEFINE_FWK_INPUT_SOURCE(PoolRASource);
DEFINE_FWK_VECTOR_INPUT_SOURCE(PoolRASource);
