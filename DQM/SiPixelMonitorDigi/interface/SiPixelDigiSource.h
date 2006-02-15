#ifndef SiPixelMonitorDigi_SiPixelDigiSource_h
#define SiPixelMonitorDigi_SiPixelDigiSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorDigi
// Class  :     SiPixelDigiSource
// 
/**

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id$
//

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"

#include <boost/cstdint.hpp>

 class SiPixelDigiSource : public edm::EDAnalyzer {
    public:
       explicit SiPixelDigiSource(const edm::ParameterSet&);
       ~SiPixelDigiSource();

       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob(edm::EventSetup const&) ;
       virtual void endJob() ;

       virtual void buildStructure(edm::EventSetup const&);
       virtual void bookMEs();

    private:
       int eventNo;
       DaqMonitorBEInterface* theDMBE;
       std::map<uint32_t,SiPixelDigiModule*> thePixelStructure;
 };

#endif
