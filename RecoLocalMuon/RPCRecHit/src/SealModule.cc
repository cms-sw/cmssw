#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitProducer.h"

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAlgoFactory.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitStandardAlgo.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(RPCRecHitProducer);
DEFINE_SEAL_PLUGIN (RPCRecHitAlgoFactory, RPCRecHitStandardAlgo, "RPCRecHitStandardAlgo");
