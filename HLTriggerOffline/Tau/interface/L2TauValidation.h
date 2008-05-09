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
      edm::InputTag     mcColl_;         // input products from HLTMcInfo
      edm::InputTag     l1taus_;         //Handle to L1 Seed

      //Select if you want match or not
      int matchLevel_;
      double matchDeltaRMC_;
      double matchDeltaRL1_;
      
      //Output file
      std::string outFile_;

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
     
      bool match(const reco::Jet&,const LVColl&);//See if this Jet Is Matched
      bool matchL1(const reco::Jet&,std::vector<l1extra::L1JetParticleRef>&);//See if this Jet Is Matched to L1
};


