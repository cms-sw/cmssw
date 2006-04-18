/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/04/12 10:19:45 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAlgoFactory.h"



RPCRecHitAlgoFactory::RPCRecHitAlgoFactory() :
  seal::PluginFactory<RPCRecHitBaseAlgo*(const edm::ParameterSet&)>("RPCRecHitAlgoFactory"){}



RPCRecHitAlgoFactory::~RPCRecHitAlgoFactory(){}



RPCRecHitAlgoFactory RPCRecHitAlgoFactory::s_instance;



RPCRecHitAlgoFactory* RPCRecHitAlgoFactory::get(void) {
  return &s_instance;
}
