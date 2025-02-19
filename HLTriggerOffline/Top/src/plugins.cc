#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTriggerOffline/Top/interface/TopValidation.h"
#include "HLTriggerOffline/Top/src/TopHLTDiLeptonOfflineDQM.h"
#include "HLTriggerOffline/Top/src/TopHLTSingleLeptonDQM.h"
#include "HLTriggerOffline/Top/interface/HLTEfficiencyCalculator.h"

DEFINE_FWK_MODULE(TopValidation);
DEFINE_FWK_MODULE(TopHLTDiLeptonOfflineDQM);
DEFINE_FWK_MODULE(TopHLTSingleLeptonDQM);
DEFINE_FWK_MODULE(HLTEffCalculator);
