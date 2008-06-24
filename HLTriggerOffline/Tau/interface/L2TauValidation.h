// Original Author:  Michail Bachtis
// Created:  Sun Jan 20 20:10:02 CST 2008
// University of Wisconsin-Madison

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include <string>


//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


 typedef math::XYZTLorentzVectorD   LV;
 typedef std::vector<LV>            LVColl;




//
// class decleration
//


class L2TauValidation : public edm::EDAnalyzer {
   public:
      explicit L2TauValidation(const edm::ParameterSet&);
      ~L2TauValidation();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
   
      //Parameters to read
      edm::InputTag     l2TauInfoAssoc_; //Path to analyze
      edm::InputTag     l2Isolated_; //Path to analyze
      edm::InputTag     mcColl_;         // input products from HLTMcInfo
      edm::InputTag     met_;             //Handle to missing Et 
      //Select if you want match or not
      int matchLevel_;
      double matchDeltaRMC_;
      
      //Tag to save in DQM File
      std::string triggerTag_;
      
      //Output file
      std::string outFile_;
      
      //Histogram Limits
      double EtMin_;
      double EtMax_;
      int NBins_;

      //Monitor elements main
      MonitorElement* jetEt;
      MonitorElement* jetEta;
      MonitorElement* jetPhi;
      MonitorElement* ecalIsolEt;
      MonitorElement* towerIsolEt;
      MonitorElement* seedTowerEt;
      MonitorElement* clusterEtaRMS;
      MonitorElement* clusterPhiRMS;
      MonitorElement* clusterDeltaRRMS;
      MonitorElement* nClusters;
      MonitorElement* EtEffNum;
      MonitorElement* EtEffDenom;
      MonitorElement* EtEff;
      MonitorElement* MET;

      bool matchJet(const reco::Jet&,const reco::CaloJetCollection&);//See if this Jet Is Matched
      bool match(const reco::Jet&,const LVColl&);//See if this Jet Is Matched

};


