#ifndef RecoLocalMuon_DTTTrigSyncFactory_H
#define RecoLocalMuon_DTTTrigSyncFactory_H

/** \class DTTTrigSyncFactory
 *  Factory of seal plugins for TTrig syncronization during RecHit reconstruction.
 *  The plugins are concrete implementations of  DTTTrigBaseSync case class.
 *
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"

namespace edm {
  class ParameterSet;
}
class DTTTrigBaseSync;

typedef edmplugin::PluginFactory<DTTTrigBaseSync *(const edm::ParameterSet &, edm::ConsumesCollector)>
    DTTTrigSyncFactory;
#endif
