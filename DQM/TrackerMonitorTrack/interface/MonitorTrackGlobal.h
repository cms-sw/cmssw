#ifndef MonitorTrackGlobal_H
#define MonitorTrackGlobal_H
// -*- C++ -*-
//
// Package:    MonitorTrackGlobal
// Class:      MonitorTrackGlobal
// 
/**\class MonitorTrackGlobal MonitorTrackGlobal.cc DQM/TrackerMonitorTrack/src/MonitorTrackGlobal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Israel Goitom
//         Created:  Tue May 23 18:35:30 CEST 2006
// $Id: MonitorTrackGlobal.h,v 1.2 2006/05/25 18:22:03 dkcira Exp $
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
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

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

  DaqMonitorBEInterface * dbe;
  edm::ParameterSet conf_;

  MonitorElement * trackSize;
  MonitorElement * recHitSize;

  MonitorElement * d0VsTheta;
  MonitorElement * d0VsPhi;
  MonitorElement * d0VsEta;
  MonitorElement * z0VsTheta;
  MonitorElement * z0VsPhi;
  MonitorElement * z0VsEta;

  MonitorElement * chiSqrd;
  MonitorElement * chiSqrdVsTheta;
  MonitorElement * chiSqrdVsPhi;
  MonitorElement * chiSqrdVsEta;

};
#endif
