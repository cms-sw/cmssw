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
// $Id: TrackingMonitor.h,v 1.2 2009/06/11 18:15:03 boudoul Exp $

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

class DQMStore;
class TrackAnalyzer;
class TProfile;

class TrackingMonitor : public edm::EDAnalyzer {
   public:
      explicit TrackingMonitor(const edm::ParameterSet&);
      ~TrackingMonitor();
      virtual void beginJob(void);
      virtual void endJob(void);

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:
  void doProfileX(TH2 * th2, MonitorElement* me);
  void doProfileX(MonitorElement * th2m, MonitorElement* me);
  

      // ----------member data ---------------------------

//  unsigned int minTracks_;

  std::string histname;  //for naming the histograms according to algorithm used

  DQMStore * dqmStore_;
  edm::ParameterSet conf_;

  // the track analyzer
  TrackAnalyzer * theTrackAnalyzer;
  edm::InputTag bsSrc;
  MonitorElement * NumberOfTracks;
  MonitorElement * NumberOfSeeds;
  MonitorElement * SeedEta;
  MonitorElement * SeedPhi;
  MonitorElement * SeedTheta;
  MonitorElement * NumberOfTrackCandidates;
  MonitorElement * NumberOfMeanRecHitsPerTrack;
  MonitorElement * NumberOfMeanLayersPerTrack;  

  std::string builderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
};
#endif
