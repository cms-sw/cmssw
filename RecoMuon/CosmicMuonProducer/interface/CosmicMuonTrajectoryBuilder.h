#ifndef CosmicMuonTrajectoryBuilder_H
#define CosmicMuonTrajectoryBuilder_H
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: 2008/02/11 16:48:34 $
 *  $Revision: 1.16 $
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
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonSmoother.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;
class CosmicMuonUtilities;
class DirectMuonNavigation;

typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

class CosmicMuonTrajectoryBuilder : public MuonTrajectoryBuilder{
public:

  /// Constructor 
  CosmicMuonTrajectoryBuilder(const edm::ParameterSet&,const MuonServiceProxy* service);

  /// Destructor
  virtual ~CosmicMuonTrajectoryBuilder();

  /// build trajectories from seed
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

  CosmicMuonSmoother* smoother() const {return theSmoother;}

  CosmicMuonUtilities* utilities() const {return smoother()->utilities();}

  DirectMuonNavigation* navigation() const {return theNavigation;}

private:

  MuonTransientTrackingRecHit::MuonRecHitContainer unusedHits(const DetLayer*, const TrajectoryMeasurement&) const;

  void explore(Trajectory&, MuonTransientTrackingRecHit::MuonRecHitContainer&);

  void buildSecondHalf(Trajectory&);

  void build(const TrajectoryStateOnSurface&, const NavigationDirection&, Trajectory&);

  TrajectoryStateOnSurface intermediateState(const TrajectoryStateOnSurface&) const;

  void selectHits(MuonTransientTrackingRecHit::MuonRecHitContainer&) const;

  /// reverse a trajectory without refit
  void reverseTrajectory(Trajectory&) const;

  void reverseTrajectoryDirection(Trajectory&) const;

  double computeNDOF(const Trajectory&) const;

  /// check if the trajectory iterates the same hit more than once
  bool selfDuplicate(const Trajectory&) const;

  /// check the direction of trajectory by refitting from both ends
  void estimateDirection(Trajectory&) const;

  void updateTrajectory(Trajectory&, const MuonTransientTrackingRecHit::MuonRecHitContainer&);

  std::vector<TrajectoryMeasurement> findBestMeasurements(const DetLayer*, const TrajectoryStateOnSurface&, const MeasurementEstimator*);

  void incrementChamberCounters(const DetLayer *layer, int& dtChambers, int& cscChambers, int& rpcChambers, int& totalChambers);

  DirectMuonNavigation* theNavigation;
  edm::ParameterSet theNavigationPSet;

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
