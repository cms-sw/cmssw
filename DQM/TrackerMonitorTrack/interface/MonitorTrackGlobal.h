#ifndef MonitorTrackGlobal_H
#define MonitorTrackGlobal_H
// -*- C++ -*-
//
// Package:    MonitorTrackGlobal
// Class:      MonitorTrackGlobal
// 
/**\class MonitorTrackGlobal MonitorTrackGlobal.cc DQM/TrackerMonitorTrack/src/MonitorTrackGlobal.cc
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Israel Goitom
//         Created:  Tue May 23 18:35:30 CEST 2006
// $Id: MonitorTrackGlobal.h,v 1.8 2007/05/19 10:35:57 dkcira Exp $

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

class MonitorTrackGlobal : public edm::EDAnalyzer {
   public:
      explicit MonitorTrackGlobal(const edm::ParameterSet&);
      ~MonitorTrackGlobal();
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
