#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Sources/interface/VectorInputSourceMacros.h"
#include "RNTupleTempSource.h"
#include "EmbeddedRNTupleTempSource.h"

using edm::rntuple_temp::EmbeddedRNTupleTempSource;
using edm::rntuple_temp::RNTupleTempSource;
DEFINE_FWK_INPUT_SOURCE(RNTupleTempSource);
DEFINE_FWK_VECTOR_INPUT_SOURCE(EmbeddedRNTupleTempSource);
