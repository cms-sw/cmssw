#ifndef RecoMTD_TimingTools_TrackPathLength_h
#define RecoMTD_TimingTools_TrackPathLength_h

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoMTD/TimingTools/interface/TrackSegments.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"

class Propagator;

namespace mtd {

  /// Compute total track path length from the PCA to the beam line to the outermost
  /// tracker hit, accumulating per-layer segments into trs.  The TSCBL must already
  /// have been computed by the caller.  Returns false if any propagation step fails.
  bool trackPathLength(const Trajectory& traj,
                       const TrajectoryStateClosestToBeamLine& tscbl,
                       const Propagator* propagator,
                       float& pathLength,
                       TrackSegments& trs);

  /// Overload that computes the TSCBL internally from the beam spot.
  /// Returns false if the TSCBL cannot be built or any propagation step fails.
  bool trackPathLength(const Trajectory& traj,
                       const reco::BeamSpot& bs,
                       const Propagator* propagator,
                       float& pathLength,
                       TrackSegments& trs);

}  // namespace mtd

#endif
