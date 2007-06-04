#ifndef MinPtTrajectoryFilter_H
#define MinPtTrajectoryFilter_H

#include "RecoTracker/CkfPattern/interface/TrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/CkfPattern/interface/TempTrajectory.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */
class MinPtTrajectoryFilter : public TrajectoryFilter {
public:

  explicit MinPtTrajectoryFilter( double ptMin, float nSigma = 5.F) : 
    thePtMin( ptMin), theNSigma(nSigma) {}


  virtual bool operator()( const Trajectory& traj) const { return test(traj.lastMeasurement(),traj.foundHits()); }
  virtual bool operator()( const TempTrajectory& traj) const { return test(traj.lastMeasurement(),traj.foundHits()); }

  virtual std::string name() const {return "MinPtTrajectoryFilter";}

private:

  bool test( const TrajectoryMeasurement & tm, int foundHits) const ;

  float thePtMin;
  float theNSigma;

};

#endif
