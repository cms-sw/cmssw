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
// $Id: SiStripMonitorDigi.h,v 1.1 2006/02/09 19:08:42 gbruno Exp $
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
       struct ModMEs{
            MonitorElement* DigisPerModule;
            MonitorElement* ADCsHottestStrip;
            MonitorElement* ADCsCoolestStrip;
       };
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       // uint32_t me_type: 1=#digis/module; 2=adcs of hottest strip/module; 3= adcs of coolest strips/module.
       std::map<uint32_t, ModMEs> DigiMEs;
};

#endif
