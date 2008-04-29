#ifndef _ClusterShapeTrajectoryFilter_h_
#define _ClusterShapeTrajectoryFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#define MaxSize 20

#include <vector>

class SiPixelRecHit;
class SiStripRecHit2D;
class TrackingGeometry;
class MagneticField;

class ClusterShapeTrajectoryFilter : public TrajectoryFilter {
  public:
    ClusterShapeTrajectoryFilter
//      (const edm::ParameterSet & pset);
      (const TrackingGeometry* theTracker_,
       const MagneticField* theMagneticField_);
//       const int theMode);

    virtual bool qualityFilter(const TempTrajectory&) const;
    virtual bool qualityFilter(const Trajectory&) const;
 
    virtual bool toBeContinued(TempTrajectory&) const;
    virtual bool toBeContinued(Trajectory&) const;

    virtual std::string name() const { return "ClusterShapeTrajectoryFilter"; }

  private:
    bool processHit(const GlobalVector gdir, const SiPixelRecHit* recHit) const;
    bool processHit(const GlobalVector gdir, const SiStripRecHit2D* recHit) const;
    bool isInside(const float a[2][2], std::pair<float,float> movement) const;

    const TrackingGeometry* theTracker;
    const MagneticField* theMagneticField;
//    const int theMode;
    float pixelLimits[2][MaxSize + 1][MaxSize + 1][2][2][2],
          stripLimits[26][2];
};

#endif
