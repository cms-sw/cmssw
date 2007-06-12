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

#include "RecoTracker/NuclearSeedGenerator/interface/TangentHelix.h"

#include <boost/shared_ptr.hpp>

class FreeTrajectoryState;

class SeedFromNuclearInteraction {
private :
  typedef TrajectoryMeasurement                       TM;
  typedef TrajectoryStateOnSurface                    TSOS;
  typedef edm::OwnVector<TrackingRecHit>              recHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer>             ConstRecHitContainer;

public :
  SeedFromNuclearInteraction(const edm::EventSetup& es, const edm::ParameterSet& iConfig);
 
  /// Fill all data members from 2 TM's where the first one is supposed to be at the interaction point
  void setMeasurements(const TM& tmAtInteractionPoint, const TM& newTM);

  /// Fill all data members from 2 TM's using the circle associated to the primary track as constraint
  void setMeasurements(const TangentHelix& primHelix, const TM& tmAtInteractionPoint, const TM& newTM);

  PTrajectoryStateOnDet trajectoryState() const { return *pTraj; }

  FreeTrajectoryState stateWithError() const;

  FreeTrajectoryState stateWithError(const TangentHelix& helix) const;

  PropagationDirection direction() const { return alongMomentum; }

  recHitContainer hits() const; 

  TrajectorySeed TrajSeed() const { return TrajectorySeed(trajectoryState(),hits(),direction()); }
 
  bool isValid() const { return isValid_; }

  const FreeTrajectoryState& updatedState() const { return freeTS; }

  const TM* outerMeasurement() const { return outerTM; }

private :
  FreeTrajectoryState                      freeTS; 
  const TM*                                innerTM;
  const TM*                                outerTM;
  
  bool                                     isValid_;

  ConstRecHitContainer                     theHits;

  ConstRecHitPointer                       outerHit;

  boost::shared_ptr<PTrajectoryStateOnDet> pTraj;

  const Propagator*                              thePropagator;
  edm::ESHandle<TrackerGeometry>                 pDD;

  bool construct();

  double rescaleDirectionFactor; /**< Rescale the direction error */
  double rescalePositionFactor;  /**< Rescale the position error */
  double rescaleCurvatureFactor; /**< Rescale the curvature error */
};
#endif
