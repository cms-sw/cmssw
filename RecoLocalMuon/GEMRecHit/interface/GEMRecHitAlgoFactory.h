#ifndef RecoLocalMuon_GEMRecHit_GEMRecHitAlgoFactory_H
#define RecoLocalMuon_GEMRecHit_GEMRecHitAlgoFactory_H
/** \class GEMRecHitAlgoFactory
 *  Factory of seal plugins for 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of GEMRecHitBaseAlgo base class.
 *
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"

typedef edmplugin::PluginFactory<GEMRecHitBaseAlgo *(const edm::ParameterSet &)> GEMRecHitAlgoFactory;
#endif
