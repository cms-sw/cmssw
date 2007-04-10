#ifndef RecoLocalMuon_RPCRecHitAlgoFactory_H
#define RecoLocalMuon_RPCRecHitAlgoFactory_H

/** \class RPCRecHitAlgoFactory
 *  Factory of seal plugins for 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of RPCRecHitBaseAlgo base class.
 *
 *  $Date: 2006/04/18 16:28:00 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"

class RPCRecHitAlgoFactory : public seal::PluginFactory<RPCRecHitBaseAlgo*(const edm::ParameterSet&)> {
public:
  /// Constructor
  RPCRecHitAlgoFactory();

  /// Destructor
  virtual ~RPCRecHitAlgoFactory();

  // Operations
  static RPCRecHitAlgoFactory* get(void);

  
private:
  static RPCRecHitAlgoFactory s_instance;
};
#endif




