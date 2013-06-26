#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/TrigXMonitor/interface/HLTScalers.h" 
DEFINE_FWK_MODULE(HLTScalers);

#include "DQM/TrigXMonitor/interface/L1Scalers.h" 
DEFINE_FWK_MODULE(L1Scalers);

#include "DQM/TrigXMonitor/interface/L1TScalersSCAL.h" 
DEFINE_FWK_MODULE(L1TScalersSCAL);

#include "DQM/TrigXMonitor/interface/HLTSeedL1LogicScalers.h" 
DEFINE_FWK_MODULE(HLTSeedL1LogicScalers);

