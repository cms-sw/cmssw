#ifndef TrackEfficiencyMonitor_H
#define TrackEfficiencyMonitor_H
// -*- C++ -*-
//
// Package:    TrackEfficiencyMonitor
// Class:      TrackEfficiencyMonitor
// 
/**\class TrackEfficiencyMonitor TrackEfficiencyMonitor.cc DQM/TrackerMonitorTrack/src/TrackEfficiencyMonitor.cc
Monitoring source to measure the track efficiency
*/
//  Original Author:  Jeremy Andrea
// Insertion in DQM:  Anne-Catherine Le Bihan
//          Created:  Thu 28 22:45:30 CEST 2008

#include <memory>
#include <fstream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"



 
 

namespace reco{class TransientTrack;}

class DQMStore;

class TrackEfficiencyMonitor : public DQMEDAnalyzer {
   public:
      typedef reco::Track Track;
      typedef reco::TrackCollection TrackCollection;
      explicit TrackEfficiencyMonitor(const edm::ParameterSet&);
      ~TrackEfficiencyMonitor();
      virtual void beginJob(void);
      virtual void endJob(void);
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

      enum SemiCylinder{Up,Down};
      std::pair<TrajectoryStateOnSurface, const DetLayer*>  findNextLayer( TrajectoryStateOnSurface startTSOS, const std::vector< const DetLayer*>& trackCompatibleLayers , bool isUpMuon   );
      SemiCylinder checkSemiCylinder(const Track&);
      void testTrackerTracks(edm::Handle<TrackCollection> tkTracks, edm::Handle<TrackCollection> staTracks);
      void testSTATracks(edm::Handle<TrackCollection> tkTracks, edm::Handle<TrackCollection> staTracks);
      bool trackerAcceptance( TrajectoryStateOnSurface theTSOS, double theRadius, double theMaxZ );
      int  compatibleLayers( TrajectoryStateOnSurface theTSOS );
  
  

   private:

 // ----------member data ---------------------------


  std::string histname;  //for naming the histograms 
  
  DQMStore * dqmStore_;
  edm::ParameterSet conf_;
  
  double theRadius_;
  double theMaxZ_;
  bool isBFieldOff_;
  bool trackEfficiency_;//1 if one wants to measure the tracking efficiency
                        //0 if one wants to measure the muon reco efficiency
		     
  edm::InputTag theTKTracksLabel_;
  edm::InputTag theSTATracksLabel_;
  edm::EDGetTokenT<reco::TrackCollection> theTKTracksToken_;
  edm::EDGetTokenT<reco::TrackCollection> theSTATracksToken_;


  int failedToPropagate;
  int nCompatibleLayers;
  bool findDetLayer;

  MonitorElement * muonX  ; 
  MonitorElement * muonY  ;
  MonitorElement * muonZ  ; 
  MonitorElement * muonEta; 
  MonitorElement * muonPhi;
  MonitorElement * muonD0;
  MonitorElement * muonCompatibleLayers;

  MonitorElement * trackX  ; 
  MonitorElement * trackY  ;
  MonitorElement * trackZ  ; 
  MonitorElement * trackEta; 
  MonitorElement * trackPhi;
  MonitorElement * trackD0;
  MonitorElement * trackCompatibleLayers;
 
  MonitorElement * deltaX    ;
  MonitorElement * deltaY    ;
  MonitorElement * signDeltaX;
  MonitorElement * signDeltaY;
  
  const DirectTrackerNavigation* theNavigation;  
  MuonServiceProxy *theMuonServiceProxy;
  
  edm::ESHandle<GeometricSearchTracker> theGeometricSearchTracker;  
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<Propagator> thePropagatorCyl;
  edm::ESHandle<TransientTrackBuilder> theTTrackBuilder;
  edm::ESHandle<GeometricSearchTracker> theTracker;
  edm::ESHandle<MagneticField> bField; 
  
  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  
};
#endif
