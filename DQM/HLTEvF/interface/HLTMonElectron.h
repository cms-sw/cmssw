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
// $Id: HLTMonElectron.h,v 1.5 2009/10/15 11:31:28 fwyzard Exp $
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

class HLTMonElectron : public edm::EDAnalyzer {
   public:
      explicit HLTMonElectron(const edm::ParameterSet&);
      ~HLTMonElectron();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& , const edm::Event&  ,unsigned int);

      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe;
      std::vector<MonitorElement *> etahist;
      std::vector<MonitorElement *> phihist;
      std::vector<MonitorElement *> ethist;
      std::vector<MonitorElement *> etahistiso;
      std::vector<MonitorElement *> phihistiso;
      std::vector<MonitorElement *> ethistiso;
      MonitorElement* total;
      std::vector<edm::InputTag> theHLTCollectionLabels;  
      std::vector<int> theHLTOutputTypes;
      std::vector<bool> plotiso;
      std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
      std::vector<std::pair<double,double> > plotBounds; 
      unsigned int reqNum;
 
      double thePtMin ;
      double thePtMax ;
      unsigned int theNbins ;
      
      std::string dirname_;
      bool monitorDaemon_;
      ofstream logFile_;
      int theHLTOutputType;
      std::string outputFile_;
      
};
#endif
