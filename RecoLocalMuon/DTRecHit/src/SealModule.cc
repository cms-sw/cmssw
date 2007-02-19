#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/DTRecHit/src/DTRecHitProducer.h"

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTRecHit/src/DTLinearDriftAlgo.h"
#include "RecoLocalMuon/DTRecHit/src/DTLinearDriftFromDBAlgo.h"
#include "RecoLocalMuon/DTRecHit/src/DTParametrizedDriftAlgo.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/src/DTTTrigSyncTOFCorr.h"
#include "CalibMuon/DTDigiSync/src/DTTTrigSyncT0Only.h"
#include "CalibMuon/DTDigiSync/src/DTTTrigSyncFromDB.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTRecHitProducer);
DEFINE_SEAL_PLUGIN (DTRecHitAlgoFactory, DTLinearDriftAlgo, "DTLinearDriftAlgo");
DEFINE_SEAL_PLUGIN (DTRecHitAlgoFactory, DTLinearDriftFromDBAlgo, "DTLinearDriftFromDBAlgo");
DEFINE_SEAL_PLUGIN (DTRecHitAlgoFactory, DTParametrizedDriftAlgo, "DTParametrizedDriftAlgo");
DEFINE_SEAL_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncTOFCorr, "DTTTrigSyncTOFCorr");
DEFINE_SEAL_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncT0Only, "DTTTrigSyncT0Only");
DEFINE_SEAL_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncFromDB, "DTTTrigSyncFromDB");
