#ifndef RecoMuon_TrackerSeedGenerator_TSGFromPropagation_H
#define RecoMuon_TrackerSeedGenerator_TSGFromPropagation_H

/** \class TSGFromPropagation
 *  Tracker Seed Generator by propagating and updating a standAlone muon
 *  to the first 2 (or 1) rechits it meets in tracker system 
 *
 *  $Date: 2013/01/08 17:08:25 $
 *  $Revision: 1.15 $
 *  \author Chang Liu - Purdue University 
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class LayerMeasurements;
class Chi2MeasurementEstimator;
class Propagator;
class MeasurementTracker;
class GeometricSearchTracker;
class DirectTrackerNavigation;
class TrajectoryStateTransform;
class TrackerTopology;

class TSGFromPropagation : public TrackerSeedGenerator {

public:
  /// constructor
  TSGFromPropagation(const edm::ParameterSet &pset);

  TSGFromPropagation(const edm::ParameterSet& par, const MuonServiceProxy*);

  /// destructor
  virtual ~TSGFromPropagation();

  /// generate seed(s) for a track
  void  trackerSeeds(const TrackCand&, const TrackingRegion&, const TrackerTopology *, std::vector<TrajectorySeed>&);
    
  /// initialize
  void init(const MuonServiceProxy*);

  /// set an event
  void setEvent(const edm::Event&);

private:

  TrajectoryStateOnSurface innerState(const TrackCand&) const;

  TrajectoryStateOnSurface outerTkState(const TrackCand&) const;

  const LayerMeasurements* tkLayerMeasurements() const { return theTkLayerMeasurements; } 

  const TrajectoryStateUpdator* updator() const {return theUpdator;}

  const Chi2MeasurementEstimator* estimator() const { return theEstimator; }

  edm::ESHandle<Propagator> propagator() const {return theService->propagator(thePropagatorName); }

  /// create a hitless seed from a trajectory state
  TrajectorySeed createSeed(const TrajectoryStateOnSurface&, const DetId&) const;

  /// create a seed from a trajectory state
  TrajectorySeed createSeed(const TrajectoryStateOnSurface& tsos, const edm::OwnVector<TrackingRecHit>& container, const DetId& id) const;

  /// select valid measurements
  void validMeasurements(std::vector<TrajectoryMeasurement>&) const;

  /// look for measurements on the first compatible layer (faster way)
  std::vector<TrajectoryMeasurement> findMeasurements_new(const DetLayer*, const TrajectoryStateOnSurface&) const;

  /// look for measurements on the first compatible layer
  std::vector<TrajectoryMeasurement> findMeasurements(const DetLayer*, const TrajectoryStateOnSurface&) const;

  /// check some quantity and beam-spot compatibility and decide to continue
  bool passSelection(const TrajectoryStateOnSurface&) const;

  void getRescalingFactor(const TrackCand& staMuon);

  /// adjust the error matrix of the FTS
  void adjust(FreeTrajectoryState &) const;

  /// adjust the error matrix of the TSOS
  void adjust(TrajectoryStateOnSurface &) const;

  double dxyDis(const TrajectoryStateOnSurface& tsos) const;

  double zDis(const TrajectoryStateOnSurface& tsos) const;

  struct increasingEstimate{
    bool operator()(const TrajectoryMeasurement& lhs,
                    const TrajectoryMeasurement& rhs) const{ 
      return lhs.estimate() < rhs.estimate();
    }
  };

  struct isInvalid {
    bool operator()(const TrajectoryMeasurement& measurement) {
      return ( ((measurement).recHit() == 0) || !((measurement).recHit()->isValid()) || !((measurement).updatedState().isValid()) ); 
    }
  };

  unsigned long long theCacheId_MT;
  unsigned long long theCacheId_TG;

  std::string theCategory;

  const LayerMeasurements*  theTkLayerMeasurements;

  edm::ESHandle<GeometricSearchTracker> theTracker;

  std::string theMeasTrackerName;
  edm::ESHandle<MeasurementTracker> theMeasTracker;

  const DirectTrackerNavigation* theNavigation;

  const MuonServiceProxy* theService;

  const TrajectoryStateUpdator* theUpdator;

  const Chi2MeasurementEstimator* theEstimator;

  TrajectoryStateTransform* theTSTransformer;

  double theMaxChi2;

  double theFlexErrorRescaling;

  double theFixedErrorRescaling;

  bool theUseVertexStateFlag;

  bool theUpdateStateFlag;

  std::string theResetMethod; 

  bool theSelectStateFlag;

  std::string thePropagatorName;

  MuonErrorMatrix * theErrorMatrixAdjuster;

  bool theAdjustAtIp;

  double theSigmaZ; 

  edm::ParameterSet theConfig;

  edm::Handle<reco::BeamSpot> beamSpot;
  edm::InputTag theBeamSpotInputTag;

};

#endif 
