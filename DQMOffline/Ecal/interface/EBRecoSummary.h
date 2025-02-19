#ifndef EBRecoSummary_h
#define EBRecoSummary_h

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


class EBRecoSummary : public edm::EDAnalyzer {
  
      public:
         explicit EBRecoSummary(const edm::ParameterSet&);
	 ~EBRecoSummary();
  
  
      private:
	 virtual void beginJob() ;
	 virtual void analyze(const edm::Event&, const edm::EventSetup&);
	 virtual void endJob() ;

	 // ----------additional functions-------------------
	 void convxtalid(int & , int &);
	 int diff_neta_s(int,int);
	 int diff_nphi_s(int,int);

         
      // DQM Store -------------------
      DQMStore* dqmStore_;
      
      std::string prefixME_;
  
      // Monitor Elements (ex THXD)
            
      // ReducedRecHits ----------------------------------------------
      // ... barrel 
      MonitorElement* h_redRecHits_EB_recoFlag;
         
      // RecHits -----------------------------------------------------
      // ... barrel 
      MonitorElement* h_recHits_EB_energyMax;
      MonitorElement* h_recHits_EB_Chi2;
      MonitorElement* h_recHits_EB_time;
      MonitorElement* h_recHits_EB_E1oE4; 
      MonitorElement* h_recHits_EB_recoFlag;
      
      // Basic Clusters ----------------------------------------------
      MonitorElement* h_basicClusters_recHits_EB_recoFlag;

      // Super Clusters ----------------------------------------------
      // ... barrel
      MonitorElement* h_superClusters_EB_nBC;
      MonitorElement* h_superClusters_EB_E1oE4;
      
      MonitorElement* h_superClusters_eta;
      MonitorElement* h_superClusters_EB_phi;
      
      protected:

	 // ----------member data ---------------------------
	 edm::InputTag recHitCollection_EB_;
         edm::InputTag redRecHitCollection_EB_;
         edm::InputTag basicClusterCollection_EB_;
	 edm::InputTag superClusterCollection_EB_;
	 
	 double ethrEB_;

	 double scEtThrEB_;

};


#endif
