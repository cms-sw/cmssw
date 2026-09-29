#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "HLTriggerOffline/Scouting/interface/Run3ScoutingPhotonDQMVariables.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using Run3ScoutingPhotonGenericDQMSource = GenericObjectDQMSource<Run3ScoutingPhoton>;

DEFINE_FWK_MODULE(Run3ScoutingPhotonGenericDQMSource);
