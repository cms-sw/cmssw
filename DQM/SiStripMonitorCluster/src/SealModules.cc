#include "PluginManager/ModuleDef.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorHLT.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorFilter.h"
#include "DQM/SiStripMonitorCluster/interface/MonitorLTC.h"
#include "FWCore/Framework/interface/MakerMacros.h"



DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripMonitorCluster);
DEFINE_ANOTHER_FWK_MODULE(SiStripMonitorHLT);
DEFINE_ANOTHER_FWK_MODULE(SiStripMonitorFilter);
DEFINE_ANOTHER_FWK_MODULE(MonitorLTC);
