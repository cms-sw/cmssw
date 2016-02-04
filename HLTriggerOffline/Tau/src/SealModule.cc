#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "HLTriggerOffline/Tau/interface/HLTTauRefCombiner.h"
#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/L2TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/L1TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/HLTTauMCProducer.h"
#include "HLTriggerOffline/Tau/interface/HLTTauRelvalQTester.h"



DEFINE_FWK_MODULE(HLTTauRefCombiner);
DEFINE_FWK_MODULE(L2TauAnalyzer);
DEFINE_FWK_MODULE(L1TauAnalyzer);
DEFINE_FWK_MODULE(L25TauAnalyzer);
DEFINE_FWK_MODULE(HLTTauMCProducer);
DEFINE_FWK_MODULE(HLTTauRelvalQTester);

