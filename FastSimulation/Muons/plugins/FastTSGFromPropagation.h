#ifndef FastSimulation_Muons_FastTSGFromPropagation_H
#define FastSimulation_Muons_FastTSGFromPropagation_H

/** \class FastTSGFromPropagation
 *  Tracker Seed Generator by propagating and updating a standAlone muon
 *  to the first 2 (or 1) rechits it meets in tracker system 
 *
 *  Emulate TSGFromPropagation in RecoMuon
 *
 *  \author Hwidong Yoo - Purdue University 
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"


class LayerMeasurements;
class Chi2MeasurementEstimator;
class Propagator;
class MeasurementTracker;
class GeometricSearchTracker;
class DirectTrackerNavigation;
class TrajectoryStateTransform;
class SimTrack;
class TrackerGeometry;
class TrackerTopology;

class FastTSGFromPropagation : public TrackerSeedGenerator {

public:
  /// constructor
  FastTSGFromPropagation(const edm::ParameterSet &pset);

  FastTSGFromPropagation(const edm::ParameterSet& par, const MuonServiceProxy*);

  /// destructor
  virtual ~FastTSGFromPropagation();

  /// generate seed(s) for a track
  void  trackerSeeds(const TrackCand&, const TrackingRegion&, 
		     const TrackerTopology *tTopo, std::vector<TrajectorySeed>&);
    
  /// initialize
  void init(const MuonServiceProxy*);

  /// set an event
  void setEvent(const edm::Event&);

private:
  /// A mere copy (without memory leak) of an existing tracking method
    void stateOnDet(const TrajectoryStateOnSurface& ts,
                      unsigned int detid,
		      PTrajectoryStateOnDet& pts) const;

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

  edm::ESHandle<MeasurementTracker> theMeasTracker;

  const DirectTrackerNavigation* theNavigation;
  const TrackerGeometry*  theGeometry;

  const MuonServiceProxy* theService;

  const TrajectoryStateUpdator* theUpdator;

  const Chi2MeasurementEstimator* theEstimator;

  TrajectoryStateTransform* theTSTransformer;

  double theMaxChi2;

  double theFlexErrorRescaling;

  double theFixedErrorRescaling;

  bool theUseVertexStateFlag;

  bool theUpdateStateFlag;

  edm::InputTag theSimTrackCollectionLabel;
  edm::InputTag theHitProducer;

  std::string theResetMethod; 

  bool theSelectStateFlag;

  std::string thePropagatorName;

  MuonErrorMatrix * theErrorMatrixAdjuster;

  bool theAdjustAtIp;

  double theSigmaZ; 

  edm::ParameterSet theConfig;
  edm::InputTag beamSpot_;

  edm::Handle<reco::BeamSpot> theBeamSpot;
  edm::Handle<edm::SimTrackContainer> theSimTracks;
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSRecHits;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;

};

#endif 
