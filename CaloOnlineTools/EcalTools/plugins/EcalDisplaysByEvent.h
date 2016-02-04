// -*- C++ -*-
//
// Package:   EcalDisplaysByEvent 
// Class:     EcalDisplaysByEvent 
// 
/**\class EcalDisplaysByEvent EcalDisplaysByEvent.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Aug 28 5:46:22 CEST 2007
// $Id: EcalDisplaysByEvent.h,v 1.3 2010/01/16 14:46:16 hegner Exp $
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
#include "TH2F.h"
#include "TH3F.h"
#include "TGraph.h"
#include "TTree.h"
#include "TCanvas.h"


//
// class declaration
//

class EcalDisplaysByEvent : public edm::EDAnalyzer {
   public:
      explicit EcalDisplaysByEvent(const edm::ParameterSet&);
      ~EcalDisplaysByEvent();


   private:
      virtual void beginRun(edm::Run const &, edm::EventSetup const &) ;
      virtual void analyze(edm::Event const &, edm::EventSetup const &);
      virtual void endJob() ;
      std::string intToString(int num);
      std::string floatToString(float num);
      void initHists(int);
      void initEvtByEvtHists(int naiveEvtNum_, int ievt);
      void deleteEvtByEvtHists();
      void initAllEventHistos();
      enum Ecal2DHistSubDetType {
	 EB_FINE 	= 0,
	 EB_COARSE 	= 1,
	 EEM_FINE	= 2,
	 EEM_COARSE	= 3,
	 EEP_FINE	= 4,
	 EEP_COARSE	= 5
      };
      TH2F* init2DEcalHist(std::string histTypeName, int subDet);
      TH3F* init3DEcalHist(std::string histTypeName, int dubDet);
      TCanvas* init2DEcalCanvas(std::string canvasName);
      void selectHits(edm::Handle<EcalRecHitCollection> hits,
          int ievt, edm::ESHandle<CaloTopology> caloTopo);
      TGraph* selectDigi(DetId det, int ievt);
      int getEEIndex(EcalElectronicsId elecId);
      void makeHistos(edm::Handle<EBDigiCollection> ebDigis);
      void makeHistos(edm::Handle<EEDigiCollection> eeDigis);
      void makeHistos(edm::Handle<EcalRecHitCollection> hits);
      void drawHistos();
      void drawCanvas(TCanvas* canvas, TH1F* hist1, TH1F* hist2, TH1F* hist3);
      void drawCanvas(TCanvas* canvas, TH2F* hist1, TH2F* hist2, TH2F* hist3);
      void drawCanvas(TCanvas* canvas, TH3F* hist1, TH3F* hist2, TH3F* hist3);
      void drawTimingErrors(TProfile2D* profile);
      void drawEELines();

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
  bool makeDigiGraphs_;
  bool makeTimingHistos_;
  bool makeEnergyHistos_;
  bool makeOccupancyHistos_;
  double histRangeMin_;
  double histRangeMax_;
  double minTimingEnergyEB_;
  double minTimingEnergyEE_;

  std::set<EBDetId> listEBChannels;
  std::set<EEDetId> listEEChannels;
    
  int abscissa[10];
  int ordinate[10];

  static float gainRatio[3];
  static edm::Service<TFileService> fileService;

  std::vector<std::string>* names;
  std::vector<std::string>* histoCanvasNamesVector;
  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<int> seedCrys_;
  std::vector<std::string> maskedEBs_;
  std::map<int,TH1F*> FEDsAndTimingHists_;
  std::map<int,float> crysAndAmplitudesMap_;
  std::map<int,EcalDCCHeaderBlock> FEDsAndDCCHeaders_;
  std::map<std::string,int> seedFrequencyMap_;
  
  TH1F* allFedsTimingHist_;
  // For event-by-evet histos
  TH1F* timingEB_;
  TH1F* timingEEM_;
  TH1F* timingEEP_;
  TH1F* energyEB_;
  TH1F* energyEEM_;
  TH1F* energyEEP_;
  TH2F* energyMapEB_;
  TH2F* energyMapEBcoarse_;
  TH2F* energyMapEEM_;
  TH2F* energyMapEEMcoarse_;
  TH2F* energyMapEEP_;
  TH2F* energyMapEEPcoarse_;
  TH2F* recHitOccupancyEB_;
  TH2F* recHitOccupancyEBcoarse_;
  TH2F* recHitOccupancyEEM_;
  TH2F* recHitOccupancyEEMcoarse_;
  TH2F* recHitOccupancyEEP_;
  TH2F* recHitOccupancyEEPcoarse_;
  TH2F* digiOccupancyEB_;
  TH2F* digiOccupancyEBcoarse_;
  TH2F* digiOccupancyEEM_;
  TH2F* digiOccupancyEEMcoarse_;
  TH2F* digiOccupancyEEP_;
  TH2F* digiOccupancyEEPcoarse_;
  TH3F* timingMapEBCoarse_;
  TH3F* timingMapEEMCoarse_;
  TH3F* timingMapEEPCoarse_;
  TH3F* timingMapEB_;
  TH3F* timingMapEEM_;
  TH3F* timingMapEEP_;
  TCanvas* timingCanvas_;
  TCanvas* energyCanvas_;
  TCanvas* energyMapCanvas_;
  TCanvas* energyMapCoarseCanvas_;
  TCanvas* recHitOccupancyCanvas_;
  TCanvas* recHitOccupancyCoarseCanvas_;
  TCanvas* digiOccupancyCanvas_;
  TCanvas* digiOccupancyCoarseCanvas_;
  TCanvas* timingMapCoarseCanvas_;
  TCanvas* timingMapCanvas_;

  // For all-event hitos
  TH1F* timingEBAll_;
  TH1F* timingEEMAll_;
  TH1F* timingEEPAll_;
  TH1F* energyEBAll_;
  TH1F* energyEEMAll_;
  TH1F* energyEEPAll_;
  TH2F* energyMapEBAll_;
  TH2F* energyMapEBcoarseAll_;
  TH2F* energyMapEEMAll_;
  TH2F* energyMapEEMcoarseAll_;
  TH2F* energyMapEEPAll_;
  TH2F* energyMapEEPcoarseAll_;
  TH2F* recHitOccupancyEBAll_;
  TH2F* recHitOccupancyEBcoarseAll_;
  TH2F* recHitOccupancyEEMAll_;
  TH2F* recHitOccupancyEEMcoarseAll_;
  TH2F* recHitOccupancyEEPAll_;
  TH2F* recHitOccupancyEEPcoarseAll_;
  TH2F* digiOccupancyEBAll_;
  TH2F* digiOccupancyEBcoarseAll_;
  TH2F* digiOccupancyEEMAll_;
  TH2F* digiOccupancyEEMcoarseAll_;
  TH2F* digiOccupancyEEPAll_;
  TH2F* digiOccupancyEEPcoarseAll_;
  TH3F* timingMapEBCoarseAll_;
  TH3F* timingMapEEMCoarseAll_;
  TH3F* timingMapEEPCoarseAll_;
  TH3F* timingMapEBAll_;
  TH3F* timingMapEEMAll_;
  TH3F* timingMapEEPAll_;
  TCanvas* timingCanvasAll_;
  TCanvas* energyCanvasAll_;
  TCanvas* energyMapCanvasAll_;
  TCanvas* energyMapCoarseCanvasAll_;
  TCanvas* recHitOccupancyCanvasAll_;
  TCanvas* recHitOccupancyCoarseCanvasAll_;
  TCanvas* digiOccupancyCanvasAll_;
  TCanvas* digiOccupancyCoarseCanvasAll_;
  TCanvas* timingMapCoarseCanvasAll_;
  TCanvas* timingMapCanvasAll_;
  
  TTree* canvasNames_;
  TTree* histoCanvasNames_;
  EcalFedMap* fedMap_;
  const EcalElectronicsMapping* ecalElectronicsMap_;
 
  int naiveEvtNum_; 
};
