#ifndef CosmicMuonTrajectoryBuilder_H
#define CosmicMuonTrajectoryBuilder_H
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: 2006/12/31 20:39:54 $
 *  $Revision: 1.9 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;

typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

class CosmicMuonTrajectoryBuilder : public MuonTrajectoryBuilder{
public:

  CosmicMuonTrajectoryBuilder(const edm::ParameterSet&,const MuonServiceProxy* service);
  virtual ~CosmicMuonTrajectoryBuilder();

  std::vector<Trajectory*> trajectories(const TrajectorySeed&);

  /// dummy implementation, unused in this class
  virtual CandidateContainer trajectories(const TrackCand&) {
    return CandidateContainer();
  }

  virtual void setEvent(const edm::Event&);

  const Propagator* propagator() const {return &*theService->propagator(thePropagatorName);}

  //FIXME
  const Propagator* propagatorAlong() const {return &*theService->propagator("SteppingHelixPropagatorAlong");}

  const Propagator* propagatorOpposite() const {return &*theService->propagator("SteppingHelixPropagatorOpposite");}

  MuonTrajectoryUpdator* updator() const {return theUpdator;}

  MuonTrajectoryUpdator* backwardUpdator() const {return theBKUpdator;}

private:

  MuonTransientTrackingRecHit::MuonRecHitContainer unusedHits(const DetLayer*, const TrajectoryMeasurement&) const;

  void print(const MuonTransientTrackingRecHit::MuonRecHitContainer&) const;

  void print(const TransientTrackingRecHit::ConstRecHitContainer&) const;

  void explore(Trajectory&, MuonTransientTrackingRecHit::MuonRecHitContainer&);

  void buildSecondHalf(Trajectory&);

  void build(const TrajectoryStateOnSurface&, const NavigationDirection&, Trajectory&);

  TrajectoryStateOnSurface intermediateState(const TrajectoryStateOnSurface&) const;

  TrajectoryStateOnSurface stepPropagate(const TrajectoryStateOnSurface&,
                                         const ConstRecHitPointer&) const;

  void selectHits(MuonTransientTrackingRecHit::MuonRecHitContainer&) const;

  /// reverse a trajectory without refit
  void reverseTrajectory(Trajectory&) const;

  double computeNDOF(const Trajectory&) const;

  void reverseDirection(TrajectoryStateOnSurface&) const;

  /// check if the trajectory iterates the same hit more than once
  bool selfDuplicate(const Trajectory&) const;

  /// check the direction of trajectory by refitting from both ends
  void estimateDirection(Trajectory&) const;

  void updateTrajectory(Trajectory&, const MuonTransientTrackingRecHit::MuonRecHitContainer&);

  MuonTrajectoryUpdator* theUpdator;
  MuonTrajectoryUpdator* theBKUpdator;
  MuonDetLayerMeasurements* theLayerMeasurements;

  const MuonServiceProxy* theService;
  CosmicMuonSmoother* theSmoother;

  std::string thePropagatorName;
  bool theTraversingMuonFlag;

  int theNTraversing;
  int theNSuccess;
  
};
#endif
