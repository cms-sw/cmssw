#ifndef MuonTransientTrackingRecHitBreaker_H
#define MuonTransientTrackingRecHitBreaker_H

/** \class MuonTransientTrackingRecHitBreaker
 *  No description available.
 *
 *  $Date: 2008/04/24 18:14:07 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class MuonTransientTrackingRecHitBreaker {

public:

  /// takes a muon rechit and returns its sub-rechits given a certain granularity 
  static TransientTrackingRecHit::ConstRecHitContainer 
  breakInSubRecHits(TransientTrackingRecHit::ConstRecHitPointer, int granularity);

protected:

private:

};
#endif

