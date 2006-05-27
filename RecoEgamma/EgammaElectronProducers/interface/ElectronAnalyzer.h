#ifndef EgammaElectronProducers_ElectronAnalyzer_h
#define EgammaElectronProducers_ElectronAnalyzer_h
// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     ElectronAnalyzer
// 
/**\class ElectronAnalyzer ElectronAnalyzer.h RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Fri May 26 16:52:45 EDT 2006
// $Id$
//

// system include files

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "TFile.h"
#include "TH1F.h"

// forward declarations

//
// class decleration
//

class ElectronAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ElectronAnalyzer(const edm::ParameterSet&);
      ~ElectronAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      std::string fileName_;
      TFile* file_;
      TH1F* numHits_;
};

#endif
