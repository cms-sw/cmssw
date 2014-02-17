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
// $Id: EcalMipGraphs.h,v 1.9 2010/01/16 14:46:16 hegner Exp $
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
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
#include "TTree.h"


//
// class declaration
//

class EcalMipGraphs : public edm::EDAnalyzer {
   public:
      explicit EcalMipGraphs(const edm::ParameterSet&);
      ~EcalMipGraphs();


   private:
      virtual void beginRun(edm::Run const &, edm::EventSetup const &) ;
      virtual void analyze(edm::Event const &, edm::EventSetup const &);
      virtual void endJob() ;
      std::string intToString(int num);
      std::string floatToString(float num);
      void writeGraphs();
      void initHists(int);
      void selectHits(edm::Handle<EcalRecHitCollection> hits,
          int ievt, edm::ESHandle<CaloTopology> caloTopo);
      TGraph* selectDigi(DetId det, int ievt);
      int getEEIndex(EcalElectronicsId elecId);

    // ----------member data ---------------------------

  edm::InputTag EBRecHitCollection_;
  edm::InputTag EERecHitCollection_;
  edm::InputTag EBDigis_;
  edm::InputTag EEDigis_;
  edm::InputTag headerProducer_;

  edm::Handle<EBDigiCollection> EBdigisHandle;
  edm::Handle<EEDigiCollection> EEdigisHandle;

  int runNum_;
  int side_;
  double threshold_;
  double minTimingAmp_;

  std::set<EBDetId> listEBChannels;
  std::set<EEDetId> listEEChannels;
    
  int abscissa[10];
  int ordinate[10];

  static float gainRatio[3];
  static edm::Service<TFileService> fileService;

  std::vector<std::string>* names;
  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<int> seedCrys_;
  std::vector<std::string> maskedEBs_;
  std::map<int,TH1F*> FEDsAndTimingHists_;
  std::map<int,float> crysAndAmplitudesMap_;
  std::map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_;
  std::map<std::string,int> seedFrequencyMap_;
  
  TH1F* allFedsTimingHist_;
  
  TFile* file_;
  TTree* canvasNames_;
  EcalFedMap* fedMap_;
  const EcalElectronicsMapping* ecalElectronicsMap_;
 
  int naiveEvtNum_; 
};
