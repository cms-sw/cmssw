#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/MCatNLOInterface/interface/MCatNLOSource.h"
#include "GeneratorInterface/MCatNLOInterface/interface/MCatNLOFilter.h"

using edm::MCatNLOSource;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(MCatNLOSource);
DEFINE_ANOTHER_FWK_MODULE(MCatNLOFilter);
