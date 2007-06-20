#ifndef SeedFromNuclearInteraction_H
#define SeedFromNuclearInteraction_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include <boost/shared_ptr.hpp>

class SeedFromNuclearInteraction {
private :
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

public :
  SeedFromNuclearInteraction(const edm::EventSetup& es, const edm::ParameterSet& iConfig);
 
  void setMeasurements(const TM& tmAtInteractionPoint, const TM& newTM);

  PTrajectoryStateOnDet trajectoryState(){ return *pTraj; }

  TSOS stateWithError(const TSOS& state) const;

  PropagationDirection direction(){ return alongMomentum; }

  recHitContainer hits(){ return _hits; }

  TrajectorySeed TrajSeed(){ return TrajectorySeed(trajectoryState(),hits(),direction()); }
 
  bool isValid() { return isValid_; }

private :
  TSOS                updatedTSOS; 
  const TM*           theNewTM;
  recHitContainer     _hits;
  bool                isValid_;
  ConstRecHitPointer innerHit;
  ConstRecHitPointer outerHit;
  boost::shared_ptr<PTrajectoryStateOnDet> pTraj;

  edm::ESHandle<TrackerGeometry>                 tracker;
  const Propagator*                              thePropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder>  theBuilder;

  bool construct();

  double rescaleDirectionFactor;
  double rescalePositionFactor;
};
#endif
