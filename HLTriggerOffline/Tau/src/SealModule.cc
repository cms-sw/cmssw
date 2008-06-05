
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "HLTriggerOffline/Tau/interface/HLTTauMcInfo.h"
#include "HLTriggerOffline/Tau/interface/HLTTauRefInfo.h"
#include "HLTriggerOffline/Tau/interface/HLTTauRefCombiner.h"

#include "HLTriggerOffline/Tau/interface/HLTTauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/MCTauCand.h"

#include "HLTriggerOffline/Tau/interface/HLTTauL25Validation.h"
#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/L2TauValidation.h"
#include "HLTriggerOffline/Tau/interface/L2TauAnalyzer.h"
#include "HLTriggerOffline/Tau/interface/L1TauAnalyzer.h"
<<<<<<< SealModule.cc
#include "HLTriggerOffline/Tau/interface/L25TauValidation.h"

=======
#include "HLTriggerOffline/Tau/interface/HLTTauValidation.h"
>>>>>>> 1.15

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTTauMcInfo);
DEFINE_ANOTHER_FWK_MODULE(HLTTauRefInfo);
DEFINE_ANOTHER_FWK_MODULE(HLTTauRefCombiner);
DEFINE_ANOTHER_FWK_MODULE(HLTTauAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(HLTTauL25Validation);
DEFINE_ANOTHER_FWK_MODULE(L25TauAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L2TauAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L2TauValidation);
DEFINE_ANOTHER_FWK_MODULE(L1TauAnalyzer);
<<<<<<< SealModule.cc
DEFINE_ANOTHER_FWK_MODULE(L25TauValidation);
=======
DEFINE_ANOTHER_FWK_MODULE(HLTTauValidation);




>>>>>>> 1.15

