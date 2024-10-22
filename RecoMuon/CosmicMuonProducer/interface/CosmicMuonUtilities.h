#ifndef CosmicMuonProducer_CosmicMuonUtilities_H
#define CosmicMuonProducer_CosmicMuonUtilities_H

/** \file CosmicMuonUtilities
 *  contain those methods that are commonly used
 *
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class Propagator;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class Trajectory;
class TrajectoryMeasurement;

typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;

class CosmicMuonUtilities {
public:
  CosmicMuonUtilities();

  virtual ~CosmicMuonUtilities();

  std::string print(const ConstMuonRecHitContainer&) const;

  std::string print(const MuonRecHitContainer&) const;

  std::string print(const ConstRecHitContainer&) const;

  bool isTraversing(const Trajectory&) const;

  void reverseDirection(TrajectoryStateOnSurface&, const MagneticField*) const;

  TrajectoryStateOnSurface stepPropagate(const TrajectoryStateOnSurface&,
                                         const ConstRecHitPointer&,
                                         const Propagator&) const;

private:
};
#endif
