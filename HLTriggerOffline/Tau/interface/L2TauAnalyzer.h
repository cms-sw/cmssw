// Original Author:  Michail Bachtis
//         Created:  Sun Jan 20 20:10:02 CST 2008
// University of Wisconsin-Madison


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <string>
#include <TTree.h>
#include <TFile.h>

typedef math::XYZTLorentzVectorD   LV;
typedef std::vector<LV>            LVColl;

//Matching struct


struct MatchElementL2 {
  bool matched;
  double deltar;
  double mcEta;
  double mcEt;
};


class L2TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L2TauAnalyzer(const edm::ParameterSet&);
      ~L2TauAnalyzer();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      //Parameters to read
      edm::InputTag  l2TauInfoAssoc_; //Path to analyze
      edm::InputTag  l1Taus_; //Path to analyze
      edm::InputTag  l1Jets_; //Path to analyze
      std::string rootFile_;          //Output File Name
      bool IsSignal_;                 //Flag to tell the analyzer if it is signal OR QCD
      edm::InputTag mcColl_;          // input products from HLTMcInfo

      
      double matchDR_;

      int cl_Nclusters;
      float  ecalIsol_Et,towerIsol_Et,cl_etaRMS,cl_phiRMS,cl_drRMS,MCeta,MCet,seedTowerEt,JetEt,JetEta,L1et,L1eta,jetEMF; 
      TFile *l2file;//File to store the histos...
      TTree *l2tree;

      MatchElementL2 match(const reco::Jet&,const LVColl&);//See if this Jet Is Matched
      MatchElementL2 match(const reco::Jet&,const l1extra::L1JetParticleCollection&);//See if this Jet Is Matched

};


