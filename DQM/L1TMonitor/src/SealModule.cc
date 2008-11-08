#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQM/L1TMonitor/interface/L1TRCT.h>
DEFINE_ANOTHER_FWK_MODULE(L1TRCT);

#include "DQM/L1TMonitor/interface/L1TdeRCT.h"
DEFINE_FWK_MODULE(L1TdeRCT);


