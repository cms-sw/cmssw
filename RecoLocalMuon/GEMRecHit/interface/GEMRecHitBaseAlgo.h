#ifndef RecoLocalMuon_GEMRecHitBaseAlgo_H
#define RecoLocalMuon_GEMRecHitBaseAlgo_H

/** \class GEMRecHitBaseAlgo
 *  Abstract algorithmic class to compute Rec Hit
 *  form a GEM digi
 *
 *  $Date: 2013/04/24 17:16:32 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "RecoLocalMuon/GEMRecHit/src/GEMEtaPartitionMask.h"
#include "RecoLocalMuon/GEMRecHit/src/GEMMaskReClusterizer.h"

class GEMCluster;
class GEMEtaPartition;
class GEMDetId;

namespace edm {
  class ParameterSet;
  class EventSetup;
}


class GEMRecHitBaseAlgo {

 public:
  
  /// Constructor
  GEMRecHitBaseAlgo(const edm::ParameterSet& config);

  /// Destructor
  virtual ~GEMRecHitBaseAlgo();  

  /// Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup) = 0;

  /// Build all hits in the range associated to the gemId, at the 1st step.
  virtual edm::OwnVector<GEMRecHit> reconstruct(const GEMEtaPartition& roll,
						const GEMDetId& gemId,
						const GEMDigiCollection::Range& digiRange,
                                                const EtaPartitionMask& mask);

  /// standard local recHit computation
  virtual bool compute(const GEMEtaPartition& roll,
                       const GEMCluster& cl,
                       LocalPoint& Point,
                       LocalError& error) const = 0;


  /// local recHit computation accounting for track direction and 
  /// absolute position
  virtual bool compute(const GEMEtaPartition& roll,
		       const GEMCluster& cl,
                       const float& angle,
                       const GlobalPoint& globPos, 
                       LocalPoint& Point,
                       LocalError& error) const = 0;
};
#endif
