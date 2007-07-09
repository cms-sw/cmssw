#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "DQM/SiStripMonitorClient/interface/SiStripOfflineDQM.h"
#include "DQM/SiStripMonitorClient/interface/SiStripAnalyser.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripOfflineDQM);
DEFINE_ANOTHER_FWK_MODULE(SiStripAnalyser);

