#ifndef GroupedTrajCandLess_H
#define GroupedTrajCandLess_H

#include <functional>
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

/** Defines an ordering of Trajectories in terms of "goodness"
 *  The comparison is done in terms of total chi**2 / ndf plus
 *  a penalty for "lost" hits.
 */

class GroupedTrajCandLess : public std::binary_function< const Trajectory&,
			    const Trajectory&, bool>
{
public:

  GroupedTrajCandLess( float p=5, float b=0) : penalty(p), bonus(b) {}

  bool operator()( const Trajectory& a, const Trajectory& b) const {
    return score(a) < score(b);
  }

  bool operator()( const TempTrajectory& a, const TempTrajectory& b) const {
    return score(a) < score(b);
  }

private:
  float score (const Trajectory& t) const
  {
//     int ndf(-5);
//     float chi2(0.);
//     vector<TrajectoryMeasurement> measurements(t.measurements());
//     for ( vector<TrajectoryMeasurement>::const_iterator im=measurements.begin();
// 	  im!=measurements.end(); im++ ) {
//       if ( im->recHit().isValid() ) {
// 	ndf += im->recHit().dimension();
// 	chi2 += im->estimate();
//       }
//     }

//     float normChi2(0.);
//     if ( ndf>0 ) {
//       // normalise chi2 to number of (2d) hits
//       normChi2 = chi2/ndf*2;
//     }
//     else {
// //       // include bonus for found hits
// //       normChi2 = chi2 - ndf/2*penalty;
//     }
//     normChi2 -= t.foundHits()*2*b;
//     return normChi2+t.lostHits()*penalty;


    return t.chiSquared()-t.foundHits()*bonus+t.lostHits()*penalty 
      + 0.5*(1-cos(t.dPhiCacheForLoopersReconstruction()))*penalty;
  }
  float score (const TempTrajectory& t) const
  {
    return t.chiSquared()-t.foundHits()*bonus+t.lostHits()*penalty 
      + 0.5*(1-cos(t.dPhiCacheForLoopersReconstruction()))*penalty;
  }

 private:
  
  float penalty;
  float bonus;

};

#endif
