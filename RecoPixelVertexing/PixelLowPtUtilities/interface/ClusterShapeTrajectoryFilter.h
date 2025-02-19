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
  //  ClusterShapeTrajectoryFilter(const edm::EventSetup& es);

  ClusterShapeTrajectoryFilter
    (const ClusterShapeHitFilter * f):theFilter(f){}

  virtual ~ClusterShapeTrajectoryFilter();

  virtual bool qualityFilter(const TempTrajectory&) const;
  virtual bool qualityFilter(const Trajectory&) const;
 
  virtual bool toBeContinued(TempTrajectory&) const;
  virtual bool toBeContinued(Trajectory&) const;

  virtual std::string name() const { return "ClusterShapeTrajectoryFilter"; }

 private:

  const ClusterShapeHitFilter * theFilter;
};

#endif
