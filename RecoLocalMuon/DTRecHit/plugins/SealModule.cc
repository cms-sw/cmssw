#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/DTRecHit/plugins/DTRecHitProducer.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTNoDriftAlgo.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTLinearDriftAlgo.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTLinearDriftFromDBAlgo.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTParametrizedDriftAlgo.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/src/DTTTrigSyncTOFCorr.h"
#include "CalibMuon/DTDigiSync/src/DTTTrigSyncT0Only.h"
#include "CalibMuon/DTDigiSync/src/DTTTrigSyncFromDB.h"


DEFINE_FWK_MODULE(DTRecHitProducer);

DEFINE_EDM_PLUGIN (DTRecHitAlgoFactory, DTNoDriftAlgo, "DTNoDriftAlgo");
DEFINE_EDM_PLUGIN (DTRecHitAlgoFactory, DTLinearDriftAlgo, "DTLinearDriftAlgo");
DEFINE_EDM_PLUGIN (DTRecHitAlgoFactory, DTLinearDriftFromDBAlgo, "DTLinearDriftFromDBAlgo");
DEFINE_EDM_PLUGIN (DTRecHitAlgoFactory, DTParametrizedDriftAlgo, "DTParametrizedDriftAlgo");
DEFINE_EDM_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncTOFCorr, "DTTTrigSyncTOFCorr");
DEFINE_EDM_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncT0Only, "DTTTrigSyncT0Only");
DEFINE_EDM_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncFromDB, "DTTTrigSyncFromDB");
