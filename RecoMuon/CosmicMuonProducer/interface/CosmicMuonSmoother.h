#ifndef CosmicMuonProducer_CosmicMuonSmoother_H
#define CosmicMuonProducer_CosmicMuonSmoother_H

/** \file CosmicMuonSmoother
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu  -  Purdue University
 */

#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackConverter.h"

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

  virtual std::vector<Trajectory> trajectories(const Trajectory&) const;

  virtual CosmicMuonSmoother* clone() const {
    return new CosmicMuonSmoother(*this);
  }

 /// refit trajectory
    virtual TrajectoryContainer trajectories(const TrajectorySeed& seed,
				             const ConstRecHitContainer& hits, 
				             const TrajectoryStateOnSurface& firstPredTsos) const;


  const Propagator* propagator() const {return &*theService->propagator(thePropagatorName);}

  KFUpdator* updator() const {return theUpdator;}

  Chi2MeasurementEstimator* estimator() const {return theEstimator;}

private:

  std::vector<Trajectory> fit(const Trajectory&) const;
  std::vector<Trajectory> fit(const TrajectorySeed& seed,
                              const ConstRecHitContainer& hits,
                              const TrajectoryStateOnSurface& firstPredTsos) const;
  std::vector<Trajectory> smooth(const std::vector<Trajectory>& ) const;
  std::vector<Trajectory> smooth(const Trajectory&) const;


  void print(const ConstMuonRecHitContainer&) const;

  void print(const ConstRecHitContainer&) const;

  void reverseDirection(TrajectoryStateOnSurface&) const;

  TrajectoryStateOnSurface stepPropagate(const TrajectoryStateOnSurface&,
                                         const ConstRecHitPointer&) const;

  KFUpdator* theUpdator;
  Chi2MeasurementEstimator* theEstimator;

  const MuonServiceProxy* theService;

  std::string thePropagatorName;
  
};
#endif
