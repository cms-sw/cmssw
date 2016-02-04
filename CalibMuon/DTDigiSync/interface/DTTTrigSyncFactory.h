#ifndef RecoLocalMuon_DTTTrigSyncFactory_H
#define RecoLocalMuon_DTTTrigSyncFactory_H

/** \class DTTTrigSyncFactory
 *  Factory of seal plugins for TTrig syncronization during RecHit reconstruction.
 *  The plugins are concrete implementations of  DTTTrigBaseSync case class.
 *
 *  $Date: 2007/04/17 22:46:21 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
}
class DTTTrigBaseSync;

typedef edmplugin::PluginFactory<DTTTrigBaseSync *(const edm::ParameterSet &)> DTTTrigSyncFactory;
#endif

