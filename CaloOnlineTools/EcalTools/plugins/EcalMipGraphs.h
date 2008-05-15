// -*- C++ -*-
//
// Package:   EcalMipGraphs 
// Class:     EcalMipGraphs 
// 
/**\class EcalMipGraphs EcalMipGraphs.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalMipGraphs.cc,v 1.11 2007/12/19 14:32:12 franzoni Exp $
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

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TNtuple.h"


//
// class declaration
//

class EcalMipGraphs : public edm::EDAnalyzer {
   public:
      explicit EcalMipGraphs(const edm::ParameterSet&);
      ~EcalMipGraphs();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::string intToString(int num);
      void writeGraphs();

    // ----------member data ---------------------------

  edm::InputTag EcalUncalibratedRecHitCollection_;
  edm::InputTag EBDigis_;
  int runNum_;
  int side_;
  int givenSeedCry_;
  double threshold_;
  std::string fileName_;

  std::set<EBDetId> listAllChannels;
    
  int abscissa[10];
  int ordinate[10];
  
  std::vector<TGraph> graphs;
  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<std::string> maskedEBs_;

  TFile* file;
  TNtuple* eventsAndSeedCrys_;
  EcalFedMap* fedMap;
  
};
