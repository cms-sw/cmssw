#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/RCTMonitor/src/checkRCT.h"
#include "DQM/RCTMonitor/src/makeEfficiencyPlots.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(checkRCT);
DEFINE_ANOTHER_FWK_MODULE(makeEfficiencyPlots);
