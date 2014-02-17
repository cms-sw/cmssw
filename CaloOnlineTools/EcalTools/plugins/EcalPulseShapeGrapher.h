// -*- C++ -*-
//
// Package:    EcalPulseShapeGrapher
// Class:      EcalPulseShapeGrapher
// 
/**\class EcalPulseShapeGrapher EcalPulseShapeGrapher.cc EcalPulseShapeGrapher.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth Cooper
//         Created:  Tue Feb  5 11:35:45 CST 2008
// $Id: EcalPulseShapeGrapher.h,v 1.2 2010/01/04 15:07:40 ferriff Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TH1F.h"
#include <vector>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

//
// class decleration
//

class EcalPulseShapeGrapher : public edm::EDAnalyzer {
   public:
      explicit EcalPulseShapeGrapher(const edm::ParameterSet&);
      ~EcalPulseShapeGrapher();


   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      std::string intToString(int);

      edm::InputTag EBUncalibratedRecHitCollection_;
      edm::InputTag EBDigis_;
      edm::InputTag EEUncalibratedRecHitCollection_;
      edm::InputTag EEDigis_;

      int abscissa[10];
      int ordinate[10];
      std::vector<int> listChannels_;
      std::map<int,TH1F*> ampHistMap_;
      std::map<int,TH2F*> pulseShapeHistMap_;
      std::map<int,TH1F*> firstSampleHistMap_;
      std::map<int,TH2F*> rawPulseShapeHistMap_;
      std::map<int,TH1F*> cutAmpHistMap_;
      
      int ampCut_;
      std::string rootFilename_;

      TFile* file_;
         
      EcalFedMap* fedMap_;
      // ----------member data ---------------------------
};
