#ifndef RecoLocalMuon_RPCRecHitStandardAlgo_H
#define RecoLocalMuon_RPCRecHitStandardAlgo_H

/** \class RPCRecHitStandardAlgo
 *  Concrete implementation of RPCRecHitBaseAlgo.
 *
 *  \author M. Maggi -- INFN Bari
 */

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"

class RPCRecHitStandardAlgo : public RPCRecHitBaseAlgo {
 public:
  /// Constructor
  RPCRecHitStandardAlgo(const edm::ParameterSet& config):RPCRecHitBaseAlgo(config) {};

  /// Destructor
  ~RPCRecHitStandardAlgo() override {};

  /// Pass the Event Setup to the algo at each event
  void setES(const edm::EventSetup& setup) override {};


  bool compute(const RPCRoll& roll,
                       const RPCCluster& cluster,
                       LocalPoint& point,
                       LocalError& error,
                       float& time, float& timeErr) const override;


  bool compute(const RPCRoll& roll,
                       const RPCCluster& cluster,
                       const float& angle,
                       const GlobalPoint& globPos,
                       LocalPoint& point,
                       LocalError& error,
                       float& time, float& timeErr) const override;
};
#endif

