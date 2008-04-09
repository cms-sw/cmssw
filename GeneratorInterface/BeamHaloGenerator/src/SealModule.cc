#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/BeamHaloGenerator/interface/BeamHaloSource.h"
#include "GeneratorInterface/BeamHaloGenerator/interface/BeamHaloProducer.h"

using edm::BeamHaloSource;
using edm::BeamHaloProducer;

DEFINE_SEAL_MODULE();
DEFINE_FWK_INPUT_SOURCE(BeamHaloSource);
DEFINE_FWK_MODULE(BeamHaloProducer);
