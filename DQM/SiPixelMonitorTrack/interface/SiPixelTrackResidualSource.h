#ifndef SiPixelTrackResidualSource_H
#define SiPixelTrackResidualSource_H

// Package: SiPixelMonitorTrack
// Class:   SiPixelTrackResidualSource
// 
// class SiPixelTrackResidualSource SiPixelTrackResidualSource.h 
//       DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualSource.h
// 
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelTrackResidualSource.h,v 1.5 2008/03/01 20:19:51 lat Exp $


#include <boost/cstdint.hpp>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualModule.h"


class SiPixelTrackResidualSource : public edm::EDAnalyzer {
  public:
    explicit SiPixelTrackResidualSource(const edm::ParameterSet&);
            ~SiPixelTrackResidualSource();

    virtual void beginJob(edm::EventSetup const& iSetup);
    virtual void endJob(void);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

  private: 
    edm::ParameterSet pSet_; 
    edm::InputTag src_; 
    DQMStore* dbe_; 

    bool debug_; 

    std::map<uint32_t, SiPixelTrackResidualModule*> theSiPixelStructure; 

    MonitorElement* meSubdetResidualX[3];
    MonitorElement* meSubdetResidualY[3];
};

#endif
