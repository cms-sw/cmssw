/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"



DTRecHitAlgoFactory::DTRecHitAlgoFactory() :
  seal::PluginFactory<DTRecHitBaseAlgo*(const edm::ParameterSet&)>("DTRecHitAlgoFactory"){}



DTRecHitAlgoFactory::~DTRecHitAlgoFactory(){}



DTRecHitAlgoFactory DTRecHitAlgoFactory::s_instance;



DTRecHitAlgoFactory* DTRecHitAlgoFactory::get(void) {
  return &s_instance;
}
