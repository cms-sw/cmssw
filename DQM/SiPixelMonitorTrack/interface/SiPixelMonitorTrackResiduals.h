#ifndef SiPixelMonitorTrackResiduals_H
#define SiPixelMonitorTrackResiduals_H

// Package: SiPixelMonitorTrack
// Class:   SiPixelMonitorTrackResiduals
// 
// class SiPixelMonitorTrackResiduals SiPixelMonitorTrackResiduals.h 
//       DQM/SiPixelMonitorTrack/interface/SiPixelMonitorTrackResiduals.h
// 
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelMonitorTrackResiduals.h,v 1.2 2007/05/24 06:11:33 schuang Exp $


#include <boost/cstdint.hpp>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelResidualModule.h"


class SiPixelMonitorTrackResiduals : public edm::EDAnalyzer {
  public:
    explicit SiPixelMonitorTrackResiduals(const edm::ParameterSet&);
            ~SiPixelMonitorTrackResiduals();

    virtual void beginJob(edm::EventSetup const& iSetup);
    virtual void endJob(void);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    DaqMonitorBEInterface* dbe_;
    edm::ParameterSet conf_; 

    std::map<uint32_t, SiPixelResidualModule*> thePixelStructure; 

    // MonitorElement* meSubpixelResidualX[3];
    // MonitorElement* meSubpixelResidualY[3];
};

#endif
