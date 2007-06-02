#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorPedestals.h"
#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorRawData.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripMonitorPedestals);
DEFINE_ANOTHER_FWK_MODULE(SiStripMonitorRawData);
