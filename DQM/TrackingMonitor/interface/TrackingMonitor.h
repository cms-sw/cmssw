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
// Original Author:  Israel Goitom (Suchandra Dutta )
//         Created:  Thu 28 22:45:30 CEST 2008
// $Id: TrackingMonitor.h,v 1.9 2008/02/28 14:53:51 dutta Exp $

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class TrackingMonitor : public edm::EDAnalyzer {
   public:
      explicit TrackingMonitor(const edm::ParameterSet&);
      ~TrackingMonitor();
      virtual void beginJob(edm::EventSetup const& iSetup);
      virtual void endJob(void);

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  std::map<uint32_t, MonitorElement *> detModules;

//  unsigned int minTracks_;

  std::string histname;  //for naming the histograms according to algorithm used

  DaqMonitorBEInterface * dbe;
  edm::ParameterSet conf_;
  bool MTCCData;

  MonitorElement * NumberOfTracks;
  MonitorElement * NumberOfRecHitsPerTrack;
  MonitorElement * NumberOfMeanRecHitsPerTrack;
  MonitorElement * NumberOfRecHitsPerTrackVsPhi;
  MonitorElement * NumberOfRecHitsPerTrackVsTheta;
  MonitorElement * NumberOfRecHitsPerTrackVsEta;

  MonitorElement * TrackPx;
  MonitorElement * TrackPy;
  MonitorElement * TrackPz;
  MonitorElement * TrackPt;

  MonitorElement * TrackPhi;
  MonitorElement * TrackEta;
  MonitorElement * TrackTheta;

  MonitorElement * DistanceOfClosestApproach;
  MonitorElement * DistanceOfClosestApproachVsTheta;
  MonitorElement * DistanceOfClosestApproachVsPhi;
  MonitorElement * DistanceOfClosestApproachVsEta;

  MonitorElement * xPointOfClosestApproach;
  MonitorElement * yPointOfClosestApproach;
  MonitorElement * zPointOfClosestApproach;

  MonitorElement * Chi2;
  MonitorElement * Chi2overDoF;
  MonitorElement * Chi2overDoFVsTheta;
  MonitorElement * Chi2overDoFVsPhi;
  MonitorElement * Chi2overDoFVsEta;

};
#endif
