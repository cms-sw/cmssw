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
// $Id: ElectronAnalyzer.h,v 1.2 2006/06/21 17:02:13 pivarski Exp $
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
      TH1F* numCand_;

      std::string mctruthProducer_;
      std::string mctruthCollection_;
      std::string superClusterProducer_;
      std::string superClusterCollection_;
      std::string electronProducer_;
      std::string electronCollection_;
      std::string siHitProducer_;
      std::string siRphiHitCollection_;
      std::string siStereoHitCollection_;
};

#endif
