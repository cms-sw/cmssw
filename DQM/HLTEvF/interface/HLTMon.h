#ifndef HLTMON_H
#define HLTMON_H
// -*- C++ -*-
//
// Package:    HLTMon
// Class:      HLTMon
// 
/**\class HLTMon HLTMon.cc DQM/HLTEvF/pulgins/HLTMonElectron.cc

 Description: This is a DQM source meant to be an example for general
 development of HLT DQM code. Based on the general structure used for 
 L1TMonitor DQM sources. This adaptation of HLTMonElectron was created
 by Ben BLOOM bbloom@gmail.com

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Wed Jan 16 15:55:28 CET 2008
// $Id: HLTMon.h,v 1.3 2009/10/15 11:31:28 fwyzard Exp $
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

class HLTMon : public edm::EDAnalyzer {
   public:
      explicit HLTMon(const edm::ParameterSet&);
      ~HLTMon();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& , const edm::Event&  ,unsigned int);

      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe;
      std::vector<MonitorElement *> etahist;
      std::vector<MonitorElement *> ethist;
      std::vector<MonitorElement *> phihist;
      std::vector<MonitorElement *> eta_phihist;
      std::vector<MonitorElement *> etahistiso;
      std::vector<MonitorElement *> ethistiso;
      std::vector<MonitorElement *> phihistiso;
      MonitorElement* total;
      std::vector<edm::InputTag> theHLTCollectionLabels;  
      std::vector<int> theHLTOutputTypes;
      std::vector<bool> plotiso;
      std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
      std::vector<std::pair<double,double> > plotBounds; 
      unsigned int reqNum;
 
      double thePtMin ;
      double thePtMax ;
      double thePtMinTemp;
      double thePtMaxTemp;
      unsigned int theNbins ;
      
      std::string dirname_;
      bool monitorDaemon_;
      ofstream logFile_;
      int theHLTOutputType;
      std::string outputFile_;

      std::string histoTitle;
      
};
#endif
