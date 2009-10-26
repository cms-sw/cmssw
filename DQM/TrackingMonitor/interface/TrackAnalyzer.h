#ifndef TrackAnalyzer_H
#define TrackAnalyzer_H
// -*- C++ -*-
//
// 
/**\class TrackingMonitor TrackingMonitor.cc DQM/TrackerMonitorTrack/src/TrackingMonitor.cc
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Suchandra Dutta, Giorgia Mila
//         Created:  Thu 28 22:45:30 CEST 2008
// $Id: TrackAnalyzer.h,v 1.3 2009/09/14 16:19:21 jengbou Exp $

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;

class TrackAnalyzer {
 public:
  TrackAnalyzer(const edm::ParameterSet&);
  virtual ~TrackAnalyzer();
  virtual void beginJob(edm::EventSetup const& iSetup,DQMStore * dqmStore_);
  
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track);
  
 private:

  void fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
  void bookHistosForState(std::string sname, DQMStore * dqmStore_);
  void doTrackerSpecificInitialization(DQMStore * dqmStore_);
  void doTrackerSpecificFillHists(const reco::Track & track);
  
  // ----------member data ---------------------------
  
//  unsigned int minTracks_;
  
  std::string histname;  //for naming the histograms according to algorithm used
  
  edm::ParameterSet conf_;

  
  MonitorElement * NumberOfRecHitsPerTrack;
  MonitorElement * NumberOfRecHitsFoundPerTrack;
  MonitorElement * NumberOfRecHitsLostPerTrack;
  MonitorElement * NumberOfTOBRecHitsPerTrack;
  MonitorElement * NumberOfTIBRecHitsPerTrack;
  MonitorElement * NumberOfTIDRecHitsPerTrack;
  MonitorElement * NumberOfTECRecHitsPerTrack;
  MonitorElement * NumberOfPixBarrelRecHitsPerTrack;
  MonitorElement * NumberOfPixEndcapRecHitsPerTrack;
  MonitorElement * NumberOfLayersPerTrack;
  MonitorElement * NumberOfTOBLayersPerTrack;
  MonitorElement * NumberOfTIBLayersPerTrack;
  MonitorElement * NumberOfTIDLayersPerTrack;
  MonitorElement * NumberOfTECLayersPerTrack;
  MonitorElement * NumberOfPixBarrelLayersPerTrack;
  MonitorElement * NumberOfPixEndcapLayersPerTrack;
  MonitorElement * Chi2;
  MonitorElement * nChi2Prob;
  MonitorElement * Chi2overDoF;
  MonitorElement * DistanceOfClosestApproach;
  MonitorElement * DistanceOfClosestApproachToBS;
  MonitorElement * DistanceOfClosestApproachVsTheta;
  MonitorElement * DistanceOfClosestApproachVsPhi;
  MonitorElement * DistanceOfClosestApproachToBSVsPhi;
  MonitorElement * DistanceOfClosestApproachVsEta;
  MonitorElement * xPointOfClosestApproach;
  MonitorElement * xPointOfClosestApproachVsZ0;
  MonitorElement * yPointOfClosestApproach;
  MonitorElement * yPointOfClosestApproachVsZ0;
  MonitorElement * zPointOfClosestApproach;
 
  
  struct TkParameterMEs {
    MonitorElement * TrackP;
    MonitorElement * TrackPx;
    MonitorElement * TrackPy;
    MonitorElement * TrackPz;
    MonitorElement * TrackPt;
    
    MonitorElement * TrackPxErr;
    MonitorElement * TrackPyErr;
    MonitorElement * TrackPzErr;
    MonitorElement * TrackPtErr;
    MonitorElement * TrackPErr;
    
    MonitorElement * TrackPhi;
    MonitorElement * TrackEta;
    MonitorElement * TrackTheta;
    
    MonitorElement * TrackPhiErr;
    MonitorElement * TrackEtaErr;
    MonitorElement * TrackThetaErr;
    
    MonitorElement * NumberOfRecHitsPerTrackVsPhi;
    MonitorElement * NumberOfRecHitsPerTrackVsTheta;
    MonitorElement * NumberOfRecHitsPerTrackVsEta;
    MonitorElement * NumberOfRecHitsPerTrackVsPhiProfile;
    MonitorElement * NumberOfRecHitsPerTrackVsThetaProfile;
    MonitorElement * NumberOfRecHitsPerTrackVsEtaProfile;
  
    MonitorElement * Chi2overDoFVsTheta;
    MonitorElement * Chi2overDoFVsPhi;
    MonitorElement * Chi2overDoFVsEta;
    
  };
  
  std::map<std::string, TkParameterMEs> TkParameterMEMap;
  
 bool createHistosForState_;
 bool doTrackerSpecific_;
 bool doAllPlots_;
 bool doBSPlots_;
 
};
#endif
