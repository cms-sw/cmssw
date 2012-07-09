#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTriggerOffline/Top/interface/TopValidation.h"
#include "DQM/Physics/src/TopDiLeptonOfflineDQM.h"
#include "DQM/Physics/src/TopSingleLeptonDQM.h"

DEFINE_FWK_MODULE(TopValidation);
DEFINE_FWK_MODULE(TopDiLeptonOfflineDQM);
DEFINE_FWK_MODULE(TopSingleLeptonDQM);

