#ifndef RecoLocalMuon_GEMRecHitStandardAlgo_H
#define RecoLocalMuon_GEMRecHitStandardAlgo_H

/** \class GEMRecHitStandardAlgo
 *  Concrete implementation of GEMRecHitBaseAlgo.
 *
 *  $Date: 2013/04/24 17:16:36 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */

#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"



class GEMRecHitStandardAlgo : public GEMRecHitBaseAlgo {
 public:
  /// Constructor
  GEMRecHitStandardAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~GEMRecHitStandardAlgo();

  // Operations

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);


  virtual bool compute(const GEMEtaPartition& roll,
                       const GEMCluster& cluster,
                       LocalPoint& point,
                       LocalError& error) const;


  virtual bool compute(const GEMEtaPartition& roll,
                       const GEMCluster& cluster,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& point,
                       LocalError& error) const;
};
#endif


