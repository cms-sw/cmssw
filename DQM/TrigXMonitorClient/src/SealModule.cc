#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/TrigXMonitorClient/interface/HLTScalersClient.h" 
DEFINE_FWK_MODULE(HLTScalersClient);
#include "DQM/TrigXMonitorClient/interface/L1ScalersClient.h" 
DEFINE_FWK_MODULE(L1ScalersClient);


