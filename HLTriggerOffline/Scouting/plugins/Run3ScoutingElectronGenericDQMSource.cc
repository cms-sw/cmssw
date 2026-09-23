#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "HLTriggerOffline/Scouting/interface/Run3ScoutingElectronDQMVariables.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using Run3ScoutingElectronGenericDQMSource = GenericObjectDQMSource<Run3ScoutingElectron>;

DEFINE_FWK_MODULE(Run3ScoutingElectronGenericDQMSource);
