#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQMServices/Examples/interface/ConverterTester.h>
DEFINE_ANOTHER_FWK_MODULE(ConverterTester);
