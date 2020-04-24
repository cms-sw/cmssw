#ifndef TransientInitialStateEstimator_H
#define TransientInitialStateEstimator_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"

#include <utility>

class Propagator;
class GeomDet;
class Trajectory;
class TrackingComponentsRecord;
namespace edm { class EventSetup;}

/// Computes the trajectory state to be used as a starting state for the track fit
/// from the vector of hits. The parameters of this state are close to the final fit parameters.
/// The error matrix is enlarged in order not to bias the track fit.

class TransientInitialStateEstimator {
public:

  typedef TrajectoryStateOnSurface TSOS;

  TransientInitialStateEstimator(const edm::ParameterSet& conf);
  void setEventSetup( const edm::EventSetup& es, const TkClonerImpl& hc );

  std::pair<TrajectoryStateOnSurface, const GeomDet*>
    innerState( const Trajectory& traj, bool doBackFit=true) const;


private:
  const std::string thePropagatorAlongName;
  const std::string thePropagatorOppositeName;
  const Propagator *thePropagatorAlong;
  const Propagator *thePropagatorOpposite; // not used? can we remove it?
  TkClonerImpl theHitCloner;
  const int theNumberMeasurementsForFit;
};

#endif
