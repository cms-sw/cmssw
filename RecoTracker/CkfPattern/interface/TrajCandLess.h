#ifndef TrajCandLess_H
#define TrajCandLess_H

#include <functional>
#include "TrackingTools/PatternTools/interface/Trajectory.h"

/** Defines an ordering of Trajectories in terms of "goodness"
 *  The comparison is done in terms of total chi**2 plus
 *  a penalty for "lost" hits.
 *  This is OK for tracks of almost same number of hits, as
 *  in TrajectoryBuilder use, but is not applicable for
 *  tracks of different number of thits: the shortest track
 *  is likely to be considered "better".
 */
template <class TR>
class TrajCandLess : public std::binary_function< const TR&,
		     const TR&, bool>
{
public:

  TrajCandLess( float p=5) : penalty(p) {}

  bool operator()( const TR& a, const TR& b) const {
    return a.chiSquared() + a.lostHits()*penalty <
      b.chiSquared() + b.lostHits()*penalty;
  }

private:

  float penalty;

};

#endif
