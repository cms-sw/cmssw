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

class TrackingRegion;
class MuonServiceProxy;

class TSGForRoadSearch : public TrackerSeedGenerator {

public:
  typedef std::vector<TrajectorySeed> BTSeedCollection;  
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;

  TSGForRoadSearch(const edm::ParameterSet &pset);

  virtual ~TSGForRoadSearch();

  void init(const MuonServiceProxy *service);
  void setEvent(const edm::Event &event);

  BTSeedCollection trackerSeeds(const TrackCand&, const TrackingRegion&);

private:
  //  virtual void run(TrajectorySeedCollection &seeds, 
  //	     const edm::Event &ev, const edm::EventSetup &es, const TrackingRegion& region);
 //concrete implementation
  void makeSeeds_0(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_1(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_2(const reco::Track &,std::vector<TrajectorySeed> &);
  void makeSeeds_3(const reco::Track &,std::vector<TrajectorySeed> &);


private:
  edm::ParameterSet theConfig;

  edm::ESHandle<MeasurementTracker> _measurementTracker;
  //  edm::ESHandle<GlobalTrackingGeometry> _glbtrackergeo;
  //edm::ESHandle<MagneticField> _field;
  const MuonServiceProxy * theProxyService;

  uint _option;
  bool _copyMuonRecHit;
  std::string _propagatorName;
  edm::ESHandle<Propagator> _prop;
  std::string _propagatorCompatibleName;
  edm::ESHandle<Propagator> _propCompatible;
  Chi2MeasurementEstimator * _chi2Estimator;
  std::string _category;

};


#endif 
