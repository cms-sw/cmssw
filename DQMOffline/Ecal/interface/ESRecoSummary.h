#ifndef ESRecoSummary_h
#define ESRecoSummary_h

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


class ESRecoSummary : public edm::EDAnalyzer {
  
      public:
         explicit ESRecoSummary(const edm::ParameterSet&);
	 ~ESRecoSummary();
  
  
      private:
	 virtual void beginJob() ;
	 virtual void analyze(const edm::Event&, const edm::EventSetup&);
	 virtual void endJob() ;

      // DQM Store -------------------
      DQMStore* dqmStore_;
  
      std::string prefixME_;

      // PRESHOWER ----------------------------------------------
      MonitorElement* h_recHits_ES_energyMax;
      MonitorElement* h_recHits_ES_time;
      
      MonitorElement* h_esClusters_energy_plane1;
      MonitorElement* h_esClusters_energy_plane2;
      MonitorElement* h_esClusters_energy_ratio;
         
      protected:
  

	 // ----------member data ---------------------------
      edm::InputTag superClusterCollection_EE_;
	 edm::InputTag esRecHitCollection_;
	 edm::InputTag esClusterCollectionX_ ;
	 edm::InputTag esClusterCollectionY_ ;
	 

};


#endif
