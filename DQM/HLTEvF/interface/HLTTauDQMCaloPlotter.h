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


class HLTTauDQMCaloPlotter {
   public:
       HLTTauDQMCaloPlotter(const edm::ParameterSet&,int,int,int,double,bool,double);
      ~HLTTauDQMCaloPlotter();
      void analyze(const edm::Event&, const edm::EventSetup&,const LVColl&);

   private:

      //Parameters to read
      edm::InputTag     l2TauInfoAssoc_; //Path to analyze
      edm::InputTag     met_;             //Handle to missing Et 
      bool              doRef_;           //DoReference Analysis
      //Select if you want match or not
      double matchDeltaRMC_;
      std::string triggerTag_;//tag for dqm flder
      edm::InputTag     l2Isolated_; //Path to analyze
      
      //Histogram Limits

      double EtMax_;
      int NPtBins_;
      int NEtaBins_;
      int NPhiBins_;


      DQMStore* store;

      //Monitor elements main
      MonitorElement* jetEt;
      MonitorElement* jetEta;
      MonitorElement* jetPhi;
      MonitorElement* jetEtRes;

      MonitorElement* isoJetEt;
      MonitorElement* isoJetEta;
      MonitorElement* isoJetPhi;

      MonitorElement* ecalIsolEt;
      MonitorElement* hcalIsolEt;

      MonitorElement* seedEcalEt;
      MonitorElement* seedHcalEt;

      MonitorElement* ecalClusterEtaRMS;
      MonitorElement* ecalClusterPhiRMS;
      MonitorElement* ecalClusterDeltaRRMS;
      MonitorElement* nEcalClusters;

      MonitorElement* hcalClusterEtaRMS;
      MonitorElement* hcalClusterPhiRMS;
      MonitorElement* hcalClusterDeltaRRMS;
      MonitorElement* nHcalClusters;



      MonitorElement* EtEffNum;
      MonitorElement* EtEffDenom;
      MonitorElement* EtaEffNum;
      MonitorElement* EtaEffDenom;
      MonitorElement* PhiEffNum;
      MonitorElement* PhiEffDenom;

      bool matchJet(const reco::Jet&,const reco::CaloJetCollection&);//See if this Jet Is Matched
      std::pair<bool,LV> match(const reco::Jet&,const LVColl&);//See if this Jet Is Matched

};


