#ifndef RecoLocalMuon_GEMRecHitAlgoFactory_H
#define RecoLocalMuon_GEMRecHitAlgoFactory_H

/** \class GEMRecHitAlgoFactory
 *  Factory of seal plugins for 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of GEMRecHitBaseAlgo base class.
 *
 *  $Date: 2013/04/24 17:16:32 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<GEMRecHitBaseAlgo *(const edm::ParameterSet &)> GEMRecHitAlgoFactory;
#endif




