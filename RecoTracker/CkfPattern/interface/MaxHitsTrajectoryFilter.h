#ifndef MaxHitsTrajectoryFilter_H
#define MaxHitsTrajectoryFilter_H

#include "RecoTracker/CkfPattern/interface/TrajectoryFilter.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class MaxHitsTrajectoryFilter : public TrajectoryFilter {
public:

  explicit MaxHitsTrajectoryFilter( int maxHits=-1) : 
    theMaxHits( maxHits) {}

  virtual bool operator()( const Trajectory&) const;

  virtual std::string name() const {return "MaxHitsTrajectoryFilter";}

private:

  float theMaxHits;

};

#endif
