#ifndef RecoLocalMuon_DTRecHitAlgoFactory_H
#define RecoLocalMuon_DTRecHitAlgoFactory_H

/** \class DTRecHitAlgoFactory
 *  Factory of seal plugins for DT 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecHitBaseAlgo base class.
 *
 *  $Date: 2007/04/10 23:14:35 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<DTRecHitBaseAlgo *(const edm::ParameterSet &)> DTRecHitAlgoFactory;
#endif




