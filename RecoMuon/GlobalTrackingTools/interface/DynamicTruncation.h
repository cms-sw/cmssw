#ifndef RecoMuon_GlobalTrackingTools_DynamicTruncation_h
#define RecoMuon_GlobalTrackingTools_DynamicTruncation_h

/**
 *  Class: DynamicTruncation
 *
 *  Description:
 *  class for the dynamical stop of the KF according to the
 *  compatibility degree between the extrapolated track
 *  state and the reconstructed segment in the muon chambers
 *
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include <memory>
#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/GlobalTrackingTools/interface/StateSegmentMatcher.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoMuon/Navigation/interface/MuonNavigableLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "DataFormats/MuonReco/interface/DYTInfo.h"
#include "RecoMuon/GlobalTrackingTools/interface/ThrParameters.h"
#include "RecoMuon/GlobalTrackingTools/interface/ChamberSegmentUtility.h"


class DynamicTruncation {
  
 public:

  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

  DynamicTruncation(const edm::Event&, const MuonServiceProxy&);

  ~DynamicTruncation();

  void setProd(const edm::Handle<DTRecSegment4DCollection>& DTSegProd, 
	       const edm::Handle<CSCSegmentCollection>& CSCSegProd) {
    getSegs->initCSU(DTSegProd, CSCSegProd);
  }

  void setSelector(int);
  void setThr(const std::vector<int>&);
  void setUpdateState(bool);
  void setUseAPE(bool);
  
  // Return the vector with the tracker plus the selected muon hits
  TransientTrackingRecHit::ConstRecHitContainer filter(const Trajectory&);

  // Return the DYTInfo object 
  reco::DYTInfo getDYTInfo() {
    dytInfo.setNStUsed(nStationsUsed);
    dytInfo.setDYTEstimators(estimatorMap);
    dytInfo.setUsedStations(usedStationMap);
    dytInfo.setIdChambers(idChamberMap);
    return dytInfo;
  }

 private:

  void                 compatibleDets(TrajectoryStateOnSurface&, std::map<int, std::vector<DetId> >&);
  void                 filteringAlgo();
  void                 fillSegmentMaps(std::map<int, std::vector<DetId> >&, std::map<int, std::vector<DTRecSegment4D> >&, std::map<int, std::vector<CSCSegment> >&);
  void                 preliminaryFit(std::map<int, std::vector<DetId> >, std::map<int, std::vector<DTRecSegment4D> >, std::map<int, std::vector<CSCSegment> >);
  bool                 chooseLayers(int&, double const &, DTRecSegment4D const &, TrajectoryStateOnSurface const &, double const &, CSCSegment const &, TrajectoryStateOnSurface const &);
  void                 fillDYTInfos(int const&, bool const&, int&, double const&, double const&, DTRecSegment4D const&, CSCSegment const&);
  int                  stationfromDet(DetId const&);
  void                 update(TrajectoryStateOnSurface&, ConstRecHitPointer);
  void                 updateWithDThits(TrajectoryStateOnSurface&, DTRecSegment4D const &);
  void                 updateWithCSChits(TrajectoryStateOnSurface&, CSCSegment const &);
  void                 getThresholdFromDB(double&, DetId const&);
  void                 correctThrByPtAndEta(double&);
  void                 getThresholdFromCFG(double&, DetId const&);
  void                 testDTstation(TrajectoryStateOnSurface&, std::vector<DTRecSegment4D> const &, double&, DTRecSegment4D&, TrajectoryStateOnSurface&);
  void                 testCSCstation(TrajectoryStateOnSurface&, std::vector<CSCSegment> const &, double&, CSCSegment&, TrajectoryStateOnSurface&);
  void                 useSegment(DTRecSegment4D const &, TrajectoryStateOnSurface const &);
  void                 useSegment(CSCSegment const &, TrajectoryStateOnSurface const &);
  void                 sort(ConstRecHitContainer&);
  
  ConstRecHitContainer result, prelFitMeas;
  bool useAPE;
  std::vector<int> Thrs;
  int nStationsUsed;
  int DYTselector;
  edm::ESHandle<Propagator> propagator;
  edm::ESHandle<Propagator> propagatorPF;
  edm::ESHandle<Propagator> propagatorCompatibleDet;
  edm::ESHandle<GlobalTrackingGeometry> theG;
  edm::ESHandle<CSCGeometry> cscGeom;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
  edm::ESHandle<TrajectoryStateUpdator> updatorHandle;
  edm::ESHandle<MuonDetLayerGeometry> navMuon;
  DirectMuonNavigation *navigation;
  edm::ESHandle<MagneticField> magfield;
  std::map<int, double> estimatorMap;
  std::map<int, bool> usedStationMap;
  std::map<int, DetId> idChamberMap;
  TrajectoryStateOnSurface currentState;
  TrajectoryStateOnSurface prelFitState;
  reco::DYTInfo dytInfo;
  std::map<DTChamberId, GlobalError> dtApeMap;
  std::map<CSCDetId, GlobalError> cscApeMap; 
  double muonPTest, muonETAest;
  const DYTThrObject* dytThresholds;
  ChamberSegmentUtility* getSegs;
  ThrParameters* thrManager;
  bool useDBforThr;
  bool doUpdateOfKFStates;
};

#endif


