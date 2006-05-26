#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackGlobal.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MonitorTrackGlobal)
DEFINE_ANOTHER_FWK_MODULE(MonitorTrackResiduals)
