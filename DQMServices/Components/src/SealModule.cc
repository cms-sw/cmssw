#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"

#include "DQMServices/Components/interface/DQMEventSource.h"

// The module providing event information 
#include "DQMServices/Components/src/EventCoordinatesSource.h"
DEFINE_FWK_MODULE(EventCoordinatesSource);

// The DQM Client input source
DEFINE_ANOTHER_FWK_INPUT_SOURCE(DQMEventSource);

// The help class running the quality tests
#include "DQMServices/Components/src/QualityTester.h"
DEFINE_ANOTHER_FWK_MODULE(QualityTester);
