#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RPCRecHitProducer.h"
#include "RPCPointProducer.h"

#include "RPCRecHitAlgoFactory.h"
#include "RPCRecHitStandardAlgo.h"

DEFINE_FWK_MODULE(RPCRecHitProducer);
DEFINE_FWK_MODULE(RPCPointProducer);
DEFINE_EDM_PLUGIN(RPCRecHitAlgoFactory, RPCRecHitStandardAlgo, "RPCRecHitStandardAlgo");
