#ifndef EERecoSummary_h
#define EERecoSummary_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// DQM includes
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// ROOT include
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TProfile2D.h"


// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitLess : public std::binary_function<EcalRecHit, EcalRecHit, bool>
{
public:
  bool operator()(EcalRecHit x, EcalRecHit y)
  {
    return (x.energy() > y.energy());
  }
};


class EERecoSummary : public edm::EDAnalyzer {
  
      public:
         explicit EERecoSummary(const edm::ParameterSet&);
	 ~EERecoSummary();
  
  
      private:
	 virtual void beginJob() ;
	 virtual void analyze(const edm::Event&, const edm::EventSetup&);
	 virtual void endJob() ;

         
      // DQM Store -------------------
      DQMStore* dqmStore_;
  
      std::string prefixME_;

      // Monitor Elements (ex THXD)
            
      // ReducedRecHits ----------------------------------------------
      // ... endcap 
      MonitorElement* h_redRecHits_EE_recoFlag;
         
      // RecHits -----------------------------------------------------
      // ... endcap
      MonitorElement* h_recHits_EE_recoFlag;
      // ... endcap +
      MonitorElement* h_recHits_EEP_energyMax;
      MonitorElement* h_recHits_EEP_Chi2;
      MonitorElement* h_recHits_EEP_time;
      // ... endcap -
      MonitorElement* h_recHits_EEM_energyMax;
      MonitorElement* h_recHits_EEM_Chi2;
      MonitorElement* h_recHits_EEM_time;

      // Basic Clusters ----------------------------------------------
      MonitorElement* h_basicClusters_recHits_EE_recoFlag;

      // Super Clusters ----------------------------------------------
      // ... endcap
      MonitorElement* h_superClusters_EEP_nBC;
      MonitorElement* h_superClusters_EEM_nBC;
      	 
      MonitorElement* h_superClusters_eta;
      MonitorElement* h_superClusters_EE_phi;
      
      protected:

	 // ----------member data ---------------------------
	 edm::InputTag recHitCollection_EE_;
         edm::InputTag redRecHitCollection_EE_;
	 edm::InputTag basicClusterCollection_EE_;
	 edm::InputTag superClusterCollection_EE_;

	 double ethrEE_;

	 double scEtThrEE_;

};


#endif
