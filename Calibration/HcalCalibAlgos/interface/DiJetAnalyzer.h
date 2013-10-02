#ifndef DiJetAnalyzer_h
#define DiJetAnalyzer_h


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

// collections
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

/*
#include "TFile.h"
#include "TTree.h"
*/

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TLorentzVector.h"

#include "Calibration/HcalCalibAlgos/src/TCell.h"


//
// class decleration
//
namespace cms{
class DiJetAnalyzer : public edm::EDAnalyzer {
   public:
      explicit DiJetAnalyzer(const edm::ParameterSet&);
      ~DiJetAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------


    edm::EDGetTokenT<reco::CaloJetCollection> tok_jets_;
    edm::EDGetTokenT<EcalRecHitCollection> tok_ec_;
    edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    edm::EDGetTokenT<HFRecHitCollection> tok_hf_;



  //  output file name with histograms
      std::string fOutputFileName ;


      TFile*      hOutputFile ;

      TTree* tree; 
  
      UInt_t  eventNumber;
      UInt_t  runNumber;
      Int_t   iEtaHit;
      UInt_t  iPhiHit;

      Float_t xTrkHcal;
      Float_t yTrkHcal;
      Float_t zTrkHcal;

      Float_t PxTrkHcal;
      Float_t PyTrkHcal;
      Float_t PzTrkHcal;

      TClonesArray* cells;

      Float_t emEnergy;
      Float_t targetE;  

      Float_t etVetoJet; 
      TLorentzVector* tagJetP4;
      TLorentzVector* probeJetP4;  
      Float_t tagJetEmFrac; 
      Float_t probeJetEmFrac; 


      bool allowMissingInputs_;

      HcalRespCorrs* oldRespCorrs; 

};
}
#endif
