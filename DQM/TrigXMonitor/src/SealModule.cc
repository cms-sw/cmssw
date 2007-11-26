#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "DQM/TrigXMonitor/interface/HLTScalers.h" 
DEFINE_ANOTHER_FWK_MODULE(HLTScalers);

