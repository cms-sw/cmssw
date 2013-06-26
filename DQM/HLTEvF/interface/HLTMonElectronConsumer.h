#ifndef HLTMONELECTRONCONSUMER_H
#define HLTMONELECTRONCONSUMER_H
// -*- C++ -*-
//
// Package:    HLTMonElectronConsumer
// Class:      HLTMonElectronConsumer
// 
/**\class HLTMonElectronConsumer HLTMonElectronConsumer.cc DQM/HLTMonElectronConsumer/src/HLTMonElectronConsumer.cc

 Description: This is a DQM source meant to be an example for general
 development of HLT DQM code. Based on the general structure used for 
 L1TMonitor DQM sources.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Wed Jan 16 15:55:28 CET 2008
// $Id: HLTMonElectronConsumer.h,v 1.2 2009/10/15 11:31:28 fwyzard Exp $
//
//


// system include files
#include <memory>
#include <unistd.h>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class HLTMonElectronConsumer : public edm::EDAnalyzer {
   public:
      explicit HLTMonElectronConsumer(const edm::ParameterSet&);
      ~HLTMonElectronConsumer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data --------------------------- 

      DQMStore * dbe;

      edm::InputTag pixeltag_;
      edm::InputTag isotag_;

      MonitorElement* isototal;
      MonitorElement* isocheck;
      MonitorElement* pixelhistosEt[4];
      MonitorElement* pixelhistosEta[4];
      MonitorElement* pixelhistosPhi[4];
      MonitorElement* pixelhistosEtOut[2];
      MonitorElement* pixelhistosEtaOut[2];
      MonitorElement* pixelhistosPhiOut[2];
      MonitorElement* pixeltotal;
      MonitorElement* pixelEff;
      MonitorElement* trackEff;

      std::string dirname_;
      std::string pixeldirname_;
      std::string isodirname_;
      bool monitorDaemon_;
      ofstream logFile_;
      std::string outputFile_;
      
};
#endif
