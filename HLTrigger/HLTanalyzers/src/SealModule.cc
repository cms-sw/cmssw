// Here are the necessary incantations to declare your module to the
// framework, so it can be referenced in a cmsRun file.
//
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTanalyzers/interface/HLTAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTAnalyzer);

