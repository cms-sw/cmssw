#ifndef GroupedTrajCandLess_H
#define GroupedTrajCandLess_H

#include <functional>
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

/** Defines an ordering of Trajectories in terms of "goodness"
 *  The comparison is done in terms of total chi**2 / ndf plus
 *  a penalty for "lost" hits.
 */

class dso_internal GroupedTrajCandLess
{
public:

  GroupedTrajCandLess( float p=5, float b=0) : penalty(p), bonus(b) {}

  template<typename T>
  bool operator()( const T& a, const T& b) const {
    return score(a) < score(b);
  }

private:
  template<typename T>
  float looperPenalty(const T&t) const {
    return (t.dPhiCacheForLoopersReconstruction()==0) ? 0.f : 
     0.5f*(1.f-std::cos(t.dPhiCacheForLoopersReconstruction()))*penalty;
  }

  template<typename T>
  float score (const T & t) const
  {
   auto bb = (t.dPhiCacheForLoopersReconstruction()==0 && t.foundHits()>8) ? 2*bonus : bonus; //extra bonus for long tracks not loopers
    if ( t.lastMeasurement().updatedState().globalMomentum().perp2() < 0.81f ) bb*=0.5f; 
    return t.chiSquared()-t.foundHits()*bb+t.lostHits()*penalty 
      + looperPenalty(t);
  }

 private:
  
  float penalty;
  float bonus;

};

#endif
