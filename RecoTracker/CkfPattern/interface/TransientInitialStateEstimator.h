#ifndef TransientInitialStateEstimator_H
#define TransientInitialStateEstimator_H

#include <utility>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


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

  TransientInitialStateEstimator( const edm::EventSetup& es, const edm::ParameterSet& conf);
  /// Call this at each event until this object will come from the EventSetup as it should
  void setEventSetup( const edm::EventSetup& es );

  std::pair<TrajectoryStateOnSurface, const GeomDet*>
    innerState( const Trajectory& traj, bool doBackFit=true) const;


private:
  std::string thePropagatorAlongName;    
  std::string thePropagatorOppositeName;  
  edm::ESWatcher<TrackingComponentsRecord> thePropagatorWatcher;
  std::unique_ptr<Propagator>  thePropagatorAlong; 
  std::unique_ptr<Propagator>  thePropagatorOpposite;
  int theNumberMeasurementsForFit;
};

#endif
