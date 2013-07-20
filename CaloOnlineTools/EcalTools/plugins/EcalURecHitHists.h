// -*- C++ -*-
//
// Package:   EcalURecHitHists 
// Class:     EcalURecHitHists 
// 
/**\class EcalURecHitHists EcalURecHitHists.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalURecHitHists.h,v 1.5 2010/01/04 15:07:40 ferriff Exp $
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

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TNtuple.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

//
// class declaration
//

class EcalURecHitHists : public edm::EDAnalyzer {
   public:
      explicit EcalURecHitHists(const edm::ParameterSet&);
      ~EcalURecHitHists();


   private:
      virtual void beginRun(edm::Run const &, edm::EventSetup const &) ;
      virtual void analyze(edm::Event const &, edm::EventSetup const &);
      virtual void endJob() ;
      std::string intToString(int num);
      void initHists(int);

    // ----------member data ---------------------------

  edm::InputTag EBUncalibratedRecHitCollection_;
  edm::InputTag EEUncalibratedRecHitCollection_;
  int runNum_;
  double histRangeMax_, histRangeMin_;
  std::string fileName_;

  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<std::string> maskedEBs_;
  std::map<int,TH1F*> FEDsAndHists_;
  std::map<int,TH1F*> FEDsAndTimingHists_;

  TH1F* allFedsHist_;
  TH1F* allFedsTimingHist_;

  TFile* file;
  EcalFedMap* fedMap_;
  const EcalElectronicsMapping* ecalElectronicsMap_;
};
