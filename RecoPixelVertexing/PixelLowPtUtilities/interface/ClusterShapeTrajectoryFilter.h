#ifndef _ClusterShapeTrajectoryFilter_h_
#define _ClusterShapeTrajectoryFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class SiPixelRecHit;
class SiStripRecHit2D;
class GlobalTrackingGeometry;
class MagneticField;
class ClusterShapeHitFilter;

class ClusterShapeTrajectoryFilter : public TrajectoryFilter {
 public:
  ClusterShapeTrajectoryFilter
    (const GlobalTrackingGeometry * theTracker_,
     const MagneticField * theMagneticField_);
  virtual ~ClusterShapeTrajectoryFilter();

  virtual bool qualityFilter(const TempTrajectory&) const;
  virtual bool qualityFilter(const Trajectory&) const;
 
  virtual bool toBeContinued(TempTrajectory&) const;
  virtual bool toBeContinued(Trajectory&) const;

  virtual std::string name() const { return "ClusterShapeTrajectoryFilter"; }

 private:
  const GlobalTrackingGeometry * theTracker;
  const MagneticField * theMagneticField;

  ClusterShapeHitFilter * theFilter;
};

#endif
