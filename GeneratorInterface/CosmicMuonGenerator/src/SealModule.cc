#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosMuoGenSource.h"

DEFINE_SEAL_MODULE();
using edm::CosMuoGenSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(CosMuoGenSource);
