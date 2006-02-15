
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTTTrigSyncFactory.h"


DTTTrigSyncFactory::DTTTrigSyncFactory() :
  seal::PluginFactory<DTTTrigBaseSync*(const edm::ParameterSet&)>("DTTTrigSyncFactory"){}



DTTTrigSyncFactory::~DTTTrigSyncFactory(){}



DTTTrigSyncFactory DTTTrigSyncFactory::s_instance;



DTTTrigSyncFactory* DTTTrigSyncFactory::get(void) {
  return &s_instance;
}
