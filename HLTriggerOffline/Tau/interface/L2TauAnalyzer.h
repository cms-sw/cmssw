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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include <string>
#include <TTree.h>
#include <TFile.h>


 typedef math::XYZTLorentzVectorD   LV;
 typedef std::vector<LV>            LVColl;

//Matching struct
struct MatchElement {
  bool matched;
  double deltar;
  double mcEta;
  double mcEt;
};


//
// class decleration
//


class L2TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L2TauAnalyzer(const edm::ParameterSet&);
      ~L2TauAnalyzer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      //Parameters to read
      edm::InputTag  l2TauInfoAssoc_; //Path to analyze
      std::string rootFile_;//Output File Name

      // std::string mcColl_;//Matched Collection
      bool IsSignal_;//Flag to tell the analyzer if it is signal OR QCD
      edm::InputTag     mcColl_; // input products from HLTMcInfo
      edm::InputTag     genJets_; //Handle to generated Jets 
      edm::InputTag     l1taus_; //Handle to L1 Taus


      //Stuff to be stored (arrays)

      int cl_Nclusters,matchBit,matchL1Bit;
      float  ecalIsol_Et,towerIsol_Et,cl_etaRMS,cl_phiRMS,cl_drRMS,MCeta,MCet,seedTowerEt,JetEt; 

      TFile *l2file;//File to store the histos...
      TTree *l2tree;


      MatchElement match(const reco::Jet&,const LVColl&);//See if this Jet Is Matched
      MatchElement matchQCD(const reco::Jet&,const reco::GenJetCollection&);//See if this Jet Is Matched
      bool matchL1(const reco::Jet&,std::vector<l1extra::L1JetParticleRef>&);//See if this Jet Is Matched to L1
      

};


