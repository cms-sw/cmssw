#ifndef HLTMonPhotonSource_H
#define HLTMonPhotonSource_H

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

class HLTMonPhotonSource : public edm::EDAnalyzer {
   public:
      explicit HLTMonPhotonSource(const edm::ParameterSet&);
      ~HLTMonPhotonSource();


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
      std::vector<MonitorElement *> histiso;
      MonitorElement * eventCounter;
      std::vector<edm::InputTag> theHLTCollectionLabels;  
      std::vector<int> theHLTOutputTypes;
      std::vector<bool> plotiso;
      std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
      std::vector<std::pair<double,double> > plotBounds; 
      unsigned int reqNum;
 
      double thePtMin ;
      double thePtMax ;
      unsigned int theNbins ;

      float maxEt;
      float eta;
      float phi;
      float sigmaetaeta;

      
      std::string dirname_;
      bool monitorDaemon_;
      std::ofstream logFile_;
      int theHLTOutputType;
      std::string outputFile_;
      
};
#endif
