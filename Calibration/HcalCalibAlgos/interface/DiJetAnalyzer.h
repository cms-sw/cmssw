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

#include "TFile.h"
#include "TTree.h"


//
// class decleration
//
namespace cms{
class DiJetAnalyzer : public edm::EDAnalyzer {
   public:
      explicit DiJetAnalyzer(const edm::ParameterSet&);
      ~DiJetAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      std::string  jets_;
      std::string  jet_product;
      std::string ec_;
      std::string hbhe_; 
      std::string ho_;
      std::string hf_; 

  //  output file name with histograms
     std::string fOutputFileName ;

     float et_jet_centr, phi_jet_centr, eta_jet_centr;
     float et_jet_forw, phi_jet_forw, eta_jet_forw;  

     int nHits; 
     float hitEta[5000], hitPhi[5000];

     TFile*      hOutputFile ;
     TTree * myTree;




// jets with |eta| < eta_1  are assumed to be calibrated 
     double eta_1; 

// jets with |eta| > eta_2 are used for calibration transfer
     double eta_2; 

// jet radius 
     double jet_R; 

// Et threshold for central jets taken for di-jet calibration
     double et_threshold; 

// max Et for the third jet 
     double et_veto; 

//  root output 
     int m_histoFlag; 


// energy matrix for min L3 algorithm 
     std::vector< std::vector<float> > eventMatrix; 

// vector  with energy of central jet 
     std::vector<float> energyVector; 

// number of iterations 
     int nIter; 

     std::map<DetId,int> NDetEntries;

     std::vector<DetId> did_selected; 

     bool allowMissingInputs_;
     const CaloGeometry* geo;


};
}
#endif
