#ifndef HLTMonJetMETDQMSource_H
#define HLTMonJetMETDQMSource_H
// -*- C++ -*-
//
// Package:    HLTMonJetMETDQMSource
// Class:      HLTMonJetMETDQMSource
// 
/**\class HLTMonJetMETDQMSource HLTMonJetMETDQMSource.cc DQM/HLTEvF/pulgins/HLTMonElectron.cc

 Description: This is a DQM source meant to be an example for general
 development of HLT DQM code. Based on the general structure used for 
 L1TMonitor DQM sources. This adaptation of HLTMonElectron was created
 by Ben BLOOM bbloom@gmail.com

 Implementation:
     <Notes on implementation>
*/


// system include files
#include <memory>
#include <unistd.h>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
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

class HLTMonJetMETDQMSource : public edm::EDAnalyzer {
 public:
  explicit HLTMonJetMETDQMSource(const edm::ParameterSet&);
  ~HLTMonJetMETDQMSource();
 private:
      virtual void beginJob();
      virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
      
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& , const edm::Event&  ,unsigned int);
      
      virtual void bookJetMET();
      HLTConfigProvider hltConfig_;
      std::string processname_;
      
      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe;
      bool debug_;
      bool verbose_;
      
      std::vector<MonitorElement *> etahist;
      std::vector<MonitorElement *> ethist;
      std::vector<MonitorElement *> phihist;
      std::vector<MonitorElement *> eta_phihist;
      std::vector<MonitorElement *> etahistiso;
      std::vector<MonitorElement *> ethistiso;
      std::vector<MonitorElement *> phihistiso;
      MonitorElement* total;
      MonitorElement* hist_hltL1sJet30;
      std::vector<edm::InputTag> theHLTCollectionLabels;  
      std::vector<int> theHLTOutputTypes;
      std::vector<bool> plotiso;
      std::vector<std::string> subdir_;
      std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
      std::vector<std::pair<double,double> > plotBounds; 
      unsigned int reqNum;
      
      double thePtMin ;
      double thePtMax ;
      double thePtMinTemp;
      double thePtMaxTemp;
      unsigned int theNbins ;
      unsigned int theNbinseta ;
      
      std::string dirname_;
      bool monitorDaemon_;
      ofstream logFile_;
      int theHLTOutputType;
      std::string outputFile_;
      //std::string subdir_;
      
      std::string histoTitle;
      
};
#endif
