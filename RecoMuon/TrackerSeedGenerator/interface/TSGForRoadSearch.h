#ifndef RecoMuon_TrackerSeedGenerator_TSGForRoadSearch_H
#define RecoMuon_TrackerSeedGenerator_TSGForRoadSearch_H

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"

class TrackingRegion;
class MuonServiceProxy;
class TrajectoryStateUpdator;

class TSGForRoadSearch : public TrackerSeedGenerator {

public:
  typedef std::vector<TrajectorySeed> BTSeedCollection;  
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;

  TSGForRoadSearch(const edm::ParameterSet &pset);

  virtual ~TSGForRoadSearch();

  void init(const MuonServiceProxy *service);
  void setEvent(const edm::Event &event);

  void  trackerSeeds(const TrackCand&, const TrackingRegion&, BTSeedCollection&);

private:
  //  virtual void run(TrajectorySeedCollection &seeds, 
  //	     const edm::Event &ev, const edm::EventSetup &es, const TrackingRegion& region);
 //concrete implementation
  void makeSeeds_0(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_1(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_2(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_3(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_4(const reco::Track &,std::vector<TrajectorySeed> &);

private:
  void pushTrajectorySeed(const reco::Track & muon, std::vector<DetLayer::DetWithState > & compatible, PropagationDirection direction, std::vector<TrajectorySeed>& result)const;
  edm::ParameterSet theConfig;

  edm::ESHandle<MeasurementTracker> theMeasurementTracker;
  //  edm::ESHandle<TrajectoryStateUpdator> theUpdator;
  TrajectoryStateUpdator * theUpdator;
  const MuonServiceProxy * theProxyService;

  uint theOption;
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
