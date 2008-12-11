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
// $Id: TrackingMonitor.h,v 1.5 2008/11/25 01:30:15 mwlebour Exp $

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
class TrackAnalyzer;

class TrackingMonitor : public edm::EDAnalyzer {
   public:
      explicit TrackingMonitor(const edm::ParameterSet&);
      ~TrackingMonitor();
      virtual void beginJob(edm::EventSetup const& iSetup);
      virtual void endJob(void);

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:

      // ----------member data ---------------------------

//  unsigned int minTracks_;

  std::string histname;  //for naming the histograms according to algorithm used

  DQMStore * dqmStore_;
  edm::ParameterSet conf_;

  // the track analyzer
  TrackAnalyzer * theTrackAnalyzer;

  MonitorElement * NumberOfTracks;
  MonitorElement * NumberOfMeanRecHitsPerTrack;
  MonitorElement * NumberOfMeanLayersPerTrack;

};
#endif
