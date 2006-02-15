#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/DTRecHit/src/DTRecHitProducer.h"

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTRecHit/src/DTLinearDriftAlgo.h"

#include "RecoLocalMuon/DTRecHit/interface/DTTTrigSyncFactory.h"
#include "RecoLocalMuon/DTRecHit/src/DTTTrigSyncTOFCorr.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTRecHitProducer);
DEFINE_SEAL_PLUGIN (DTRecHitAlgoFactory, DTLinearDriftAlgo, "DTLinearDriftAlgo");
DEFINE_SEAL_PLUGIN (DTTTrigSyncFactory, DTTTrigSyncTOFCorr, "DTTTrigSyncTOFCorr");
