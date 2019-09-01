#ifndef MuonTransientTrackingRecHitBreaker_H
#define MuonTransientTrackingRecHitBreaker_H

/** \class MuonTransientTrackingRecHitBreaker
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class MuonTransientTrackingRecHitBreaker {
public:
  /// takes a muon rechit and returns its sub-rechits given a certain granularity
  static TransientTrackingRecHit::ConstRecHitContainer breakInSubRecHits(TransientTrackingRecHit::ConstRecHitPointer,
                                                                         int granularity);

protected:
private:
};
#endif
