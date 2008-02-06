#ifndef HLTMONELECTRON_H
#define HLTMONELECTRON_H
// -*- C++ -*-
//
// Package:    HLTMonElectron
// Class:      HLTMonElectron
// 
/**\class HLTMonElectron HLTMonElectron.cc DQM/HLTMonElectron/src/HLTMonElectron.cc

 Description: This is a DQM source meant to be an example for general
 development of HLT DQM code. Based on the general structure used for 
 L1TMonitor DQM sources.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Wed Jan 16 15:55:28 CET 2008
// $Id$
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

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class HLTMonElectron : public edm::EDAnalyzer {
   public:
      explicit HLTMonElectron(const edm::ParameterSet&);
      ~HLTMonElectron();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& , std::vector<int>& ,int);

      // ----------member data --------------------------- 
      int nev_;
      DaqMonitorBEInterface * dbe;
      std::vector<MonitorElement *> etahist;
      std::vector<MonitorElement *> ethist;
      std::vector<edm::InputTag> theHLTCollectionLabels;  
      std::vector<int> theHLTOutputTypes;
      bool verbose_;
      bool monitorDaemon_;
      ofstream logFile_;
      int theHLTOutputType;
      std::string outputFile_;
      
};
#endif
