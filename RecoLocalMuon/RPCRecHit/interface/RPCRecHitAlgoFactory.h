#ifndef RecoLocalMuon_RPCRecHitAlgoFactory_H
#define RecoLocalMuon_RPCRecHitAlgoFactory_H

/** \class RPCRecHitAlgoFactory
 *  Factory of seal plugins for 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of RPCRecHitBaseAlgo base class.
 *
 *  $Date: 2007/04/10 23:14:37 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<RPCRecHitBaseAlgo *(const edm::ParameterSet &)> RPCRecHitAlgoFactory;
#endif




