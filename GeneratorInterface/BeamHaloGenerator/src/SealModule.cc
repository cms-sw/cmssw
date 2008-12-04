#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/BeamHaloGenerator/interface/BeamHaloSource.h"

DEFINE_SEAL_MODULE();
using edm::BeamHaloSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(BeamHaloSource);
