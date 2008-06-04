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
// $Id: EcalMipGraphs.h,v 1.5 2008/05/06 18:38:08 scooper Exp $
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

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
      std::string floatToString(float num);
      void writeGraphs();
      void initHists(int);
      void selectEBHits(edm::Handle<EcalRecHitCollection> EBhits,
          int ievt, edm::ESHandle<CaloTopology> caloTopo);
      void selectEEHits(edm::Handle<EcalRecHitCollection> EEhits,
          int ievt, edm::ESHandle<CaloTopology> caloTopo);
      void selectEBDigis(edm::Handle<EBDigiCollection> EBdigisHandle, int ievt,
          std::map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_);
      void selectEEDigis(edm::Handle<EEDigiCollection> EEdigisHandle, int ievt,
          std::map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_);
      

    // ----------member data ---------------------------

  edm::InputTag EBRecHitCollection_;
  edm::InputTag EERecHitCollection_;
  edm::InputTag EBDigis_;
  edm::InputTag EEDigis_;
  edm::InputTag headerProducer_;

  int runNum_;
  int side_;
  int givenSeedCry_;
  double threshold_;
  std::string fileName_;

  std::set<EBDetId> listEBChannels;
  std::set<EEDetId> listEEChannels;
    
  int abscissa[10];
  int ordinate[10];
  
  std::vector<TGraph> graphs;
  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<std::string> maskedEBs_;
  std::map<int,TH1F*> FEDsAndTimingHists_;
  std::map<int,float> crysAndAmplitudesMap_;
  
  TH1F* allFedsTimingHist_;
  
  TFile* file_;
  TNtuple* eventsAndSeedCrys_;
  EcalFedMap* fedMap_;
  const EcalElectronicsMapping* ecalElectronicsMap_;
 
  int naiveEvtNum_; 
  int graphCount;
};
