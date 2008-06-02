#ifndef TrackingMonitor_H
#define TrackingMonitor_H
// -*- C++ -*-
//
// Package:    TrackingMonitor
// Class:      TrackingMonitor
// 
/**\class TrackingMonitor TrackingMonitor.cc DQM/TrackerMonitorTrack/src/TrackingMonitor.cc
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Suchandra Dutta, Giorgia Mila
//         Created:  Thu 28 22:45:30 CEST 2008
// $Id: TrackingMonitor.h,v 1.2 2008/03/01 00:43:45 dutta Exp $

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

class TrackingMonitor : public edm::EDAnalyzer {
   public:
      explicit TrackingMonitor(const edm::ParameterSet&);
      ~TrackingMonitor();
      virtual void beginJob(edm::EventSetup const& iSetup);
      virtual void endJob(void);

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:

  void fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname);
  void bookHistosForState(std::string sname);

      // ----------member data ---------------------------

//  unsigned int minTracks_;

  std::string histname;  //for naming the histograms according to algorithm used

  DQMStore * dqmStore_;
  edm::ParameterSet conf_;



  MonitorElement * NumberOfTracks;
  MonitorElement * NumberOfRecHitsPerTrack;
  MonitorElement * NumberOfMeanRecHitsPerTrack;
  MonitorElement * Chi2;
  MonitorElement * Chi2overDoF;
  MonitorElement * DistanceOfClosestApproach;
  MonitorElement * DistanceOfClosestApproachVsTheta;
  MonitorElement * DistanceOfClosestApproachVsPhi;
  MonitorElement * DistanceOfClosestApproachVsEta;
  MonitorElement * xPointOfClosestApproach;
  MonitorElement * yPointOfClosestApproach;
  MonitorElement * zPointOfClosestApproach;


  struct TkParameterMEs {
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
    
    MonitorElement * Chi2overDoFVsTheta;
    MonitorElement * Chi2overDoFVsPhi;
    MonitorElement * Chi2overDoFVsEta;

  };

  std::map<std::string, TkParameterMEs> TkParameterMEMap;

  bool createHistosForState_;

};
#endif
