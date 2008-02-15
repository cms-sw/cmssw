#ifndef MonitorTrackResiduals_H
#define MonitorTrackResiduals_H

// -*- C++ -*-
//
// Package:    TrackerMonitorTrack
// Class:      MonitorTrackResiduals
// 
/**\class MonitorTrackResiduals MonitorTrackResiduals.h DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.cc
Monitoring source for track residuals on each detector module
*/
// Original Author:  Israel Goitom
//         Created:  Fri May 26 14:12:01 CEST 2006
// $Id: MonitorTrackResiduals.h,v 1.10 2007/05/19 10:35:57 dkcira Exp $
#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

typedef std::map<int, MonitorElement *> HistoClass;

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
