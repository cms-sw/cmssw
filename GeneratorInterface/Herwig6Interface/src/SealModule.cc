#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Source.h"
#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Producer.h"
#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Filter.h"

using edm::Herwig6Source;
using edm::Herwig6Producer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(Herwig6Source);
DEFINE_ANOTHER_FWK_MODULE(Herwig6Producer);
DEFINE_ANOTHER_FWK_MODULE(Herwig6Filter);
