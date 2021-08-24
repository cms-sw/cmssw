#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/DTRecHit/plugins/DTRecHitProducer.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTNoDriftAlgo.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTLinearDriftAlgo.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTLinearDriftFromDBAlgo.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTParametrizedDriftAlgo.h"

DEFINE_FWK_MODULE(DTRecHitProducer);

DEFINE_EDM_PLUGIN(DTRecHitAlgoFactory, DTNoDriftAlgo, "DTNoDriftAlgo");
DEFINE_EDM_PLUGIN(DTRecHitAlgoFactory, DTLinearDriftAlgo, "DTLinearDriftAlgo");
DEFINE_EDM_PLUGIN(DTRecHitAlgoFactory, DTLinearDriftFromDBAlgo, "DTLinearDriftFromDBAlgo");
DEFINE_EDM_PLUGIN(DTRecHitAlgoFactory, DTParametrizedDriftAlgo, "DTParametrizedDriftAlgo");
