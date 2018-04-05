#ifndef RecoLocalMuon_ME0RecHitAlgoFactory_H
#define RecoLocalMuon_ME0RecHitAlgoFactory_H

/** \class ME0RecHitAlgoFactory
 *  Factory of seal plugins for 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of ME0RecHitBaseAlgo base class.
 *
 *  $Date: 2014/02/04 10:16:32 $
 *  $Revision: 1.1 $
 *  \author M. Maggi - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<ME0RecHitBaseAlgo *(const edm::ParameterSet &)> ME0RecHitAlgoFactory;
#endif

