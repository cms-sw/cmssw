#ifndef RecoLocalMuon_GEMRecHitStandardAlgo_H
#define RecoLocalMuon_GEMRecHitStandardAlgo_H

/** \class GEMRecHitStandardAlgo
 *  Concrete implementation of GEMRecHitBaseAlgo.
 *
 *  \author M. Maggi -- INFN Bari
 */

#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"



class GEMRecHitStandardAlgo : public GEMRecHitBaseAlgo {
 public:
  /// Constructor
  GEMRecHitStandardAlgo(const edm::ParameterSet& config);

  /// Destructor
  ~GEMRecHitStandardAlgo() override;

  // Operations

  /// Pass the Event Setup to the algo at each event
  void setES(const edm::EventSetup& setup) override;


  bool compute(const GEMEtaPartition& roll,
                       const GEMCluster& cluster,
                       LocalPoint& point,
                       LocalError& error) const override;


  bool compute(const GEMEtaPartition& roll,
                       const GEMCluster& cluster,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& point,
                       LocalError& error) const override;
};
#endif


