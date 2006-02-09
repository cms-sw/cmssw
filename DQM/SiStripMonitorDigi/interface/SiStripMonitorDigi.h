#ifndef SiStripMonitorDigi_SiStripMonitorDigi_h
#define SiStripMonitorDigi_SiStripMonitorDigi_h
// -*- C++ -*-
//
// Package:     SiStripMonitorDigi
// Class  :     SiStripMonitorDigi
// 
/**\class SiStripMonitorDigi SiStripMonitorDigi.h DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  dkcira
//         Created:  Sat Feb  4 20:49:51 CET 2006
// $Id$
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"


class SiStripMonitorDigi : public edm::EDAnalyzer {
   public:
      explicit SiStripMonitorDigi(const edm::ParameterSet&);
      ~SiStripMonitorDigi();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&) ;
      virtual void endJob() ;

   private:
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
//       std::map<uint32_t, MonitorElement*> ADCsPerStrip; // ADCsPerStrip of a detector
       std::map<uint32_t, MonitorElement*> DigisPerDetector;
};

#endif
