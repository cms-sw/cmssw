#ifndef L1Trigger_PrimitiveAlgoFactory_H
#define L1Trigger_PrimitiveAlgoFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<RPCRecHitBaseAlgo *(const edm::ParameterSet &)> PrimitiveAlgoFactory;

#endif
