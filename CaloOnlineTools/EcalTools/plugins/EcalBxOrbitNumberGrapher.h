
// -*- C++ -*-
//
// Package:   EcalBxOrbitNumberGrapher 
// Class:     EcalBxOrbitNumberGrapher 
// 
/**\class EcalBxOrbitNumberGrapher EcalBxOrbitNumberGrapher.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalBxOrbitNumberGrapher.h,v 1.2 2010/01/04 15:07:39 ferriff Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"



#include "TFile.h"
#include "TH1F.h"


//
// class declaration
//

class EcalBxOrbitNumberGrapher : public edm::EDAnalyzer {
   public:
      explicit EcalBxOrbitNumberGrapher(const edm::ParameterSet&);
      ~EcalBxOrbitNumberGrapher();


   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void initHists(int);

    // ----------member data ---------------------------

 
  std::string digiProducer_;
  int runNum_;
  std::string fileName_;

  TH1F* bxnumberPlot_;
  TH1F* orbitErrorPlot_;
  TH1F* orbitErrorBxDiffPlot_;
  TH1F* numberofOrbitDiffPlot_;

  TFile* file;
  
};
