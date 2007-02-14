#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/VectorInputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FastSimulation/MaterialEffects/interface/NUSource.h"

using edm::NUSource;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(NUSource);
DEFINE_ANOTHER_FWK_VECTOR_INPUT_SOURCE(NUSource);
