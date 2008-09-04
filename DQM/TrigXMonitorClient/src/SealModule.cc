#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h" 
DEFINE_ANOTHER_FWK_MODULE(HLTScalersClient);
#include "DQM/TrigXMonitorClient/interface/L1ScalersClient.h" 
DEFINE_ANOTHER_FWK_MODULE(L1ScalersClient);


