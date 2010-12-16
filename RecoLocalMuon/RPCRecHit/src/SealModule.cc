#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitProducer.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCPointProducer.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAli.h"

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAlgoFactory.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitStandardAlgo.h"


DEFINE_FWK_MODULE(RPCRecHitProducer);
DEFINE_FWK_MODULE(RPCPointProducer);
DEFINE_FWK_MODULE(RPCRecHitAli);
DEFINE_EDM_PLUGIN (RPCRecHitAlgoFactory, RPCRecHitStandardAlgo, "RPCRecHitStandardAlgo");
