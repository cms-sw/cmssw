#ifndef CosmicMuonProducer_CosmicMuonSmoother_H
#define CosmicMuonProducer_CosmicMuonSmoother_H

/** \file CosmicMuonSmoother
 *
 *  $Date: 2012/09/01 12:44:22 $
 *  $Revision: 1.7 $
 *  \author Chang Liu  -  Purdue University
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonUtilities.h"

class Propagator;
class KFUpdator;
class MuonServiceProxy;
class Chi2MeasurementEstimator;

namespace edm {class ParameterSet; class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;

typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;


class CosmicMuonSmoother : public TrajectorySmoother {
public:


  CosmicMuonSmoother(const edm::ParameterSet&,const MuonServiceProxy* service);
  virtual ~CosmicMuonSmoother();

  Trajectory trajectory(const Trajectory&) const;

  virtual TrajectoryContainer trajectories(const Trajectory& traj) const {
     return TrajectorySmoother::trajectories(traj);
  }

  virtual CosmicMuonSmoother* clone() const {
    return new CosmicMuonSmoother(*this);
  }

 /// refit trajectory
    virtual TrajectoryContainer trajectories(const TrajectorySeed& seed,
				             const ConstRecHitContainer& hits, 
				             const TrajectoryStateOnSurface& firstPredTsos) const;


  const Propagator* propagatorAlong() const {return &*theService->propagator(thePropagatorAlongName);}

  const Propagator* propagatorOpposite() const {return &*theService->propagator(thePropagatorOppositeName);}

  KFUpdator* updator() const {return theUpdator;}

  CosmicMuonUtilities* utilities() const {return theUtilities; } 

  Chi2MeasurementEstimator* estimator() const {return theEstimator;}

  std::vector<Trajectory> fit(const Trajectory&) const;
  std::vector<Trajectory> fit(const TrajectorySeed& seed,
                              const ConstRecHitContainer& hits,
                              const TrajectoryStateOnSurface& firstPredTsos) const;

private:
  std::vector<Trajectory> smooth(const std::vector<Trajectory>& ) const;
  std::vector<Trajectory> smooth(const Trajectory&) const;

  TrajectoryStateOnSurface initialState(const Trajectory&) const;

  void sortHitsAlongMom(ConstRecHitContainer& hits, const TrajectoryStateOnSurface&) const;

  KFUpdator* theUpdator;
  Chi2MeasurementEstimator* theEstimator;
  CosmicMuonUtilities* theUtilities; 

  const MuonServiceProxy* theService;

  std::string thePropagatorAlongName;
  std::string thePropagatorOppositeName;
  double theErrorRescaling;
  std::string category_;
  
};
#endif
