#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/RCTMonitor/src/checkTPGs.h"
#include "DQM/RCTMonitor/src/makeEfficiencyPlots.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(checkTPGs);
DEFINE_ANOTHER_FWK_MODULE(makeEfficiencyPlots);
