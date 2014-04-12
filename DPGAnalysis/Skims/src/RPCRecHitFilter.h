#ifndef RPCRecHitsFilter_h
#define RPCRecHitsFilter_h

// Orso Iorio, INFN Napoli 

#include <string>
#include <map>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Run.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h" 
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h" 
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "FWCore/Framework/interface/EDFilter.h"

#include "TDirectory.h"
#include "TFile.h"
#include "TTree.h"

class RPCDetId;
class Trajectory;
class Propagator;
class GeomDet;
class TrajectoryStateOnSurface;

typedef std::vector<TrajectoryMeasurement>                  MeasurementContainer;
typedef std::pair<const GeomDet*, TrajectoryStateOnSurface> DetWithState;
typedef std::vector<Trajectory>                             Trajectories;


class RPCRecHitFilter : public edm::EDFilter {

public:

  explicit RPCRecHitFilter(const edm::ParameterSet&);
  ~RPCRecHitFilter() { }

private:

  virtual bool filter(edm::Event &, const edm::EventSetup&) override;

  std::string RPCDataLabel;
  
  int centralBX_, BXWindow_, minHits_, hitsInStations_;
  
  bool Verbose_, Debug_, Barrel_, EndcapPositive_, EndcapNegative_, cosmicsVeto_;

};

#endif // RPCRecHitsFilter_h
