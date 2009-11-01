#ifndef HLTMonElectronClient_H
#define HLTMonElectronClient_H

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

class HLTMonElectronClient : public edm::EDAnalyzer {
   public:
      explicit HLTMonElectronClient(const edm::ParameterSet&);
      ~HLTMonElectronClient();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data --------------------------- 

      DQMStore * dbe;

      edm::InputTag sourcetag_;

      MonitorElement* pixelhistosEt[4];
      MonitorElement* pixelhistosEta[4];
      MonitorElement* pixelhistosPhi[4];
      MonitorElement* pixelhistosEtOut[4];
      MonitorElement* pixelhistosEtaOut[4];
      MonitorElement* pixelhistosPhiOut[4];
      MonitorElement* eventCounter;
      MonitorElement* relFilterEff;
      MonitorElement* cumFilterEff;

      std::string dirname_;
      std::string sourcedirname_;
      bool monitorDaemon_;
      ofstream logFile_;
      std::string outputFile_;
      std::vector<edm::InputTag> theHLTCollectionLabels;
      
};
#endif
