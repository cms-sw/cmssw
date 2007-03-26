#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "DQMServices/Components/interface/DQMEventSource.h"

// The DQM Client input source
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(DQMEventSource);

