#ifndef MonitorTrackResiduals_H
#define MonitorTrackResiduals_H

// -*- C++ -*-
//
// Package:    TrackerMonitorTrack
// Class:      MonitorTrackResiduals
// 
/**\class MonitorTrackResiduals MonitorTrackResiduals.h DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Israel Goitom
//         Created:  Fri May 26 14:12:01 CEST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class decleration
//
typedef std::map<std::string, MonitorElement *> HistoClass;

class MonitorTrackResiduals : public edm::EDAnalyzer {
   public:
      explicit MonitorTrackResiduals(const edm::ParameterSet&);
      ~MonitorTrackResiduals();
      virtual void beginJob(edm::EventSetup const& iSetup);
      virtual void endJob(void);



      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
  std::map<uint32_t, MonitorElement *> detModules;

//  unsigned int minTracks_;

  DaqMonitorBEInterface * dbe;
  edm::ParameterSet conf_;

  HistoClass HitResidual;

};

#endif
