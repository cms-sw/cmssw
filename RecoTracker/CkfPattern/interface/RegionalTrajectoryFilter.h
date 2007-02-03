#ifndef RegionalTrajectoryFilter_h
#define RegionalTrajectoryFilter_h

#include "RecoTracker/CkfPattern/interface/TrajectoryFilter.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/CkfPattern/interface/MinPtTrajectoryFilter.h"

/** TrajectoryFilter checking compatibility with (the
 *  pt cut of) a tracking region. 
 */

class RegionalTrajectoryFilter : public TrajectoryFilter {
public:
  /// constructor from TrackingRegion
  RegionalTrajectoryFilter (const TrackingRegion& region) :
    thePtFilter(region.ptMin()) {}
  /// filter method
  bool operator() (const Trajectory& traj) const {
    return thePtFilter(traj);
  }
  /// name method imposed by TrajectoryFilter
  std::string name () const {return std::string("RegionalTrajectoryFilter");}

private:
  const MinPtTrajectoryFilter thePtFilter;
};
#endif

