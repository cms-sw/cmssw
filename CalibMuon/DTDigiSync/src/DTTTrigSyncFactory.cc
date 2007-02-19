
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/02/15 13:54:45 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"


DTTTrigSyncFactory::DTTTrigSyncFactory() :
  seal::PluginFactory<DTTTrigBaseSync*(const edm::ParameterSet&)>("DTTTrigSyncFactory"){}



DTTTrigSyncFactory::~DTTTrigSyncFactory(){}



DTTTrigSyncFactory DTTTrigSyncFactory::s_instance;



DTTTrigSyncFactory* DTTTrigSyncFactory::get(void) {
  return &s_instance;
}
