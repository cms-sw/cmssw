#ifndef _ClusterShapeTrajectoryFilter_h_
#define _ClusterShapeTrajectoryFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

namespace edm { class EventSetup; }

class SiPixelRecHit;
class SiStripRecHit2D;
class GlobalTrackingGeometry;
class MagneticField;
class SiPixelLorentzAngle;
class SiStripLorentzAngle;
class ClusterShapeHitFilter;

class ClusterShapeTrajectoryFilter : public TrajectoryFilter {
 public:
  ClusterShapeTrajectoryFilter(const edm::EventSetup& es);

  ClusterShapeTrajectoryFilter
    (const GlobalTrackingGeometry * theTracker_,
     const MagneticField          * theMagneticField_,
     const SiPixelLorentzAngle    * theSiPixelLorentzAngle_,
     const SiStripLorentzAngle    * theSiStripLorentzAngle_);

  virtual ~ClusterShapeTrajectoryFilter();

  virtual bool qualityFilter(const TempTrajectory&) const;
  virtual bool qualityFilter(const Trajectory&) const;
 
  virtual bool toBeContinued(TempTrajectory&) const;
  virtual bool toBeContinued(Trajectory&) const;

  virtual std::string name() const { return "ClusterShapeTrajectoryFilter"; }

 private:
//  const GlobalTrackingGeometry * theTracker;
//  const MagneticField * theMagneticField;

  const ClusterShapeHitFilter * theFilter;
};

#endif
