#include "FWCore/PluginManager/interface/ModuleDef.h"
//#include "FWCore/PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/ThePEGInterface/interface/ThePEGSource.h"

using edm::ThePEGSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(ThePEGSource);
