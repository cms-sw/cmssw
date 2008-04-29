
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "HLTriggerOffline/Tau/interface/HLTTauMcInfo.h"

#include "HLTriggerOffline/Tau/interface/HLTTauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/HLTTauL25Validation.h"
//#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/L2TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/L1TauAnalyzer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTTauMcInfo);
DEFINE_ANOTHER_FWK_MODULE(HLTTauAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HLTTauL25Validation);
//DEFINE_ANOTHER_FWK_MODULE(L25TauAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L2TauAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L1TauAnalyzer);



