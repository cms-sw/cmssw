#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "HLTriggerOffline/Scouting/interface/Run3ScoutingTrackDQMVariables.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using Run3ScoutingTrackGenericDQMSource = GenericObjectDQMSource<Run3ScoutingTrack>;

DEFINE_FWK_MODULE(Run3ScoutingTrackGenericDQMSource);
