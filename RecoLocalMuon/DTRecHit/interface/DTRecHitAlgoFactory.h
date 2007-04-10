#ifndef RecoLocalMuon_DTRecHitAlgoFactory_H
#define RecoLocalMuon_DTRecHitAlgoFactory_H

/** \class DTRecHitAlgoFactory
 *  Factory of seal plugins for DT 1D RecHit reconstruction algorithms.
 *  The plugins are concrete implementations of DTRecHitBaseAlgo base class.
 *
 *  $Date: 2006/02/15 13:54:45 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"

class DTRecHitAlgoFactory : public seal::PluginFactory<DTRecHitBaseAlgo*(const edm::ParameterSet&)> {
public:
  /// Constructor
  DTRecHitAlgoFactory();

  /// Destructor
  virtual ~DTRecHitAlgoFactory();

  // Operations
  static DTRecHitAlgoFactory* get(void);

  
private:
  static DTRecHitAlgoFactory s_instance;
};
#endif




