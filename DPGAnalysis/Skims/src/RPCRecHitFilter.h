#ifndef RPCRecHitsFilter_h
#define RPCRecHitsFilter_h

// Orso Iorio, INFN Napoli 



#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Run.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
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
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
//include "./MyHistoClassDbeNew.h"
//#include "./Tower.h"

#include<string>
#include<map>
#include<fstream>

#include "TDirectory.h"
#include "TFile.h"
#include "TTree.h"

class RPCDetId;
class Trajectory;
class Propagator;
class GeomDet;
class TrajectoryStateOnSurface;





typedef std::vector<TrajectoryMeasurement>          MeasurementContainer;
typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;
typedef std::vector<Trajectory> Trajectories;


class RPCRecHitFilter : public HLTFilter {

public:

  explicit RPCRecHitFilter(const edm::ParameterSet&);
  ~RPCRecHitFilter();
  

private:


  virtual void beginJob() ;

  virtual bool filter(edm::Event&, const edm::EventSetup&);

  virtual void endJob();

  std::string RPCDataLabel;
  
  //edm::InputTag RPCRecHits;

  
  int centralBX_, BXWindow_,minHits_, hitsInStations_;


  
  bool Verbose_,Debug_, Barrel_, EndcapPositive_, EndcapNegative_, cosmicsVeto_;

};
#endif
