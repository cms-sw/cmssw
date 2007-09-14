#ifndef PhysicsToolsAnalysisAnalyzerMinBias_h
#define PhysicsToolsAnalysisAnalyzerMinBias_h

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>
#include <map>
//#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace reco;
//
// class decleration
//
namespace cms{
class Analyzer_minbias : public edm::EDAnalyzer {
   public:
      explicit Analyzer_minbias(const edm::ParameterSet&);
      ~Analyzer_minbias();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:
  // ----------member data ---------------------------
     string fOutputFileName ;
  //   string datasetType ;
     
  // names of modules, producing object collections
  //   edm::InputTag m_tracksSrc;
  //
     TFile*      hOutputFile ;
 TH1D*  hHBHEEt;
 TH1D*  hHBHEEt_eta_1;
 TH1D*  hHBHEEt_eta_25;
 TH1D*  hHBHEEta;
 TH1D*  hHBHEPhi;

 TH1D*  hHFEt;
 TH1D*  hHFEt_eta_33;
 TH1D*  hHFEta;
 TH1D*  hHFPhi;
  
 TH1D*  hHOEt;
 TH1D*  hHOEt_eta_5;
 TH1D*  hHOEta;
 TH1D*  hHOPhi;
 TTree * myTree;
 int    mystart;
  //  
 int mydet, mysubd, depth, iphi, ieta;
 float phi,eta;
 float mom0,mom1,mom2,mom3,mom4,occup;
 float mom0_cut,mom1_cut,mom2_cut,mom3_cut,mom4_cut;
// counters
  map<DetId,double> theFillDetMap0;
  map<DetId,double> theFillDetMap1; 
  map<DetId,double> theFillDetMap2; 
  map<DetId,double> theFillDetMap3; 
  map<DetId,double> theFillDetMap4;
  
  map<DetId,double> theFillDetMap_cut0;
  map<DetId,double> theFillDetMap_cut1; 
  map<DetId,double> theFillDetMap_cut2; 
  map<DetId,double> theFillDetMap_cut3; 
  map<DetId,double> theFillDetMap_cut4;
   
// Calo geometry    
  const CaloGeometry* geo;
  std::vector<HcalDetId> theHcalId;
//  CaloMiscalibMapHcal mapHcal_;  
  std::string hcalfile_;
  std::ofstream *myout_hcal;

};
}
#endif
