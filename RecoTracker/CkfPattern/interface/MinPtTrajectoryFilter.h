#ifndef MinPtTrajectoryFilter_H
#define MinPtTrajectoryFilter_H

#include "RecoTracker/CkfPattern/interface/TrajectoryFilter.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class MinPtTrajectoryFilter : public TrajectoryFilter {
public:

  explicit MinPtTrajectoryFilter( double ptMin, float nSigma = 5.F) : 
    thePtMin( ptMin), theNSigma(nSigma) {}

  virtual bool operator()( const Trajectory&) const;

  virtual std::string name() const {return "MinPtTrajectoryFilter";}

private:

  float thePtMin;
  float theNSigma;

};

#endif
