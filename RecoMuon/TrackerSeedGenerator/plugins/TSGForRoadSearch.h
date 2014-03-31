#ifndef RecoMuon_TrackerSeedGenerator_TSGForRoadSearch_H
#define RecoMuon_TrackerSeedGenerator_TSGForRoadSearch_H

/** \class TSGForRoadSearch
 * Description: 
 * this class generates hit-less TrajectorySeed from a given Track.
 * the original error matrix of the Track is adjusted (configurable).
 * this class is principally used for muon HLT.
 * for options are available:
 * - inside-out seeds, on innermost Pixel/Strip layer.
 * - inside-out seeds, on innermost Strip layer.
 * - outside-in seeds, on outermost Strip layer.
 *
 * regular operation is one seed per track, but optionnaly, more than one seed can be madefor one track.
 *
 * \author Jean-Roch Vlimant
*/


#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include <RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h>
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class TrackingRegion;
class MuonServiceProxy;
class TrajectoryStateUpdator;
class TrackerTopology;

class TSGForRoadSearch : public TrackerSeedGenerator {

public:
  typedef std::vector<TrajectorySeed> BTSeedCollection;  
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;

  TSGForRoadSearch(const edm::ParameterSet &pset,edm::ConsumesCollector& IC);

  virtual ~TSGForRoadSearch();

  /// initialize the service
  void init(const MuonServiceProxy *service);
  /// set the event: update the MeasurementTracker
  void setEvent(const edm::Event &event);

  /// generated seed(s) for a track. the tracking region is not used.
  void  trackerSeeds(const TrackCand&, const TrackingRegion&, const TrackerTopology *, BTSeedCollection&);

private:
  //concrete implementation
  /// oseed from inside-out: innermost Strip layer
  void makeSeeds_0(const reco::Track &,std::vector<TrajectorySeed> &);
  /// not implemented
  void makeSeeds_1(const reco::Track &,std::vector<TrajectorySeed> &);
  /// not implemented
  void makeSeeds_2(const reco::Track &,std::vector<TrajectorySeed> &);
  /// outside-in: outermost Strip layer
  void makeSeeds_3(const reco::Track &,std::vector<TrajectorySeed> &);
  /// inside-out: innermost Pixel/Strip layer
  void makeSeeds_4(const reco::Track &,std::vector<TrajectorySeed> &);

private:
  /// get the FTS for a Track: adjusting the error matrix if requested
  bool IPfts(const reco::Track &, FreeTrajectoryState &);
  /// make the adjustement away from PCA state if requested
  bool notAtIPtsos(TrajectoryStateOnSurface & state);

  /// adjust the state at IP or where it is defined for the seed
  bool theAdjustAtIp;

  /// add the seed(s) to the collection of seeds
  void pushTrajectorySeed(const reco::Track & muon, std::vector<DetLayer::DetWithState > & compatible, PropagationDirection direction, std::vector<TrajectorySeed>& result)const;
  edm::ParameterSet theConfig;

  edm::ESHandle<MeasurementTracker> theMeasurementTracker;
  edm::ESHandle<GeometricSearchTracker> theGeometricSearchTracker;

  edm::InputTag theMeasurementTrackerEventTag;
  edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerEventToken;
  const MeasurementTrackerEvent * theMeasurementTrackerEvent;

  TrajectoryStateUpdator * theUpdator;
  const MuonServiceProxy * theProxyService;

  unsigned int theOption;
  bool theCopyMuonRecHit;
  bool theManySeeds;
  std::string thePropagatorName;
  edm::ESHandle<Propagator> theProp;
  std::string thePropagatorCompatibleName;
  edm::ESHandle<Propagator> thePropCompatible;
  Chi2MeasurementEstimator * theChi2Estimator;
  std::string theCategory;

  MuonErrorMatrix * theErrorMatrixAdjuster;
};


#endif 
