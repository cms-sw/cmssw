#ifndef RecoLocalMuon_RPCRecHitStandardAlgo_H
#define RecoLocalMuon_RPCRecHitStandardAlgo_H

/** \class RPCRecHitStandardAlgo
 *  Concrete implementation of RPCRecHitBaseAlgo.
 *
 *  $Date: 2006/04/18 16:28:31 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"



class RPCRecHitStandardAlgo : public RPCRecHitBaseAlgo {
 public:
  /// Constructor
  RPCRecHitStandardAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~RPCRecHitStandardAlgo();

  // Operations

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);


  virtual bool compute(const RPCRoll& roll,
                       const RPCCluster& cluster,
                       LocalPoint& point,
                       LocalError& error) const;


  virtual bool compute(const RPCRoll& roll,
                       const RPCCluster& cluster,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& point,
                       LocalError& error) const;
};
#endif


