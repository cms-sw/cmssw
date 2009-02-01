#ifndef DQMOFFLINE_ECAL_ECALOFFLINECOSMICTASK
#define DQMOFFLINE_ECAL_ECALOFFLINECOSMICTASK
// -*- C++ -*-
//
// Package:    EcalOfflineCosmicTask
// Class:      EcalOfflineCosmicTask
// 
/**\class EcalOfflineCosmicTask EcalOfflineCosmicTask.cc DQMOffline/EcalOfflineCosmicTask/src/EcalOfflineCosmicTask.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sean Patrick Lynch
//         Created:  Thu Nov  6 16:44:17 CET 2008
// $Id: EcalOfflineCosmicTask.h,v 1.2 2008/12/04 12:39:52 slynch Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

//#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
//#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
//#include "DataFormats/DetId/interface/DetId.h"
//#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

//#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

// *** for TrackAssociation
//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/Common/interface/Handle.h"
//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
//#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


//
// class decleration
//

class EcalOfflineCosmicTask : public edm::EDAnalyzer {
   public:
      explicit EcalOfflineCosmicTask(const edm::ParameterSet&);
      ~EcalOfflineCosmicTask();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // Booking Functions
      std::vector<float> makeXBins(int ecalSubDet, bool isCoarse);
      std::vector<float> makeYBins(int ecalSubDet, bool isCoarse);
      int getNumBins(std::vector<float> bins);
      void bookClusterHists(int ecalSubDet, int numBins);
      void bookEnergyHists(int ecalSubDet, int numBins);
      void bookOccupancyHists(int ecalSubDet);
      void bookOccupancyTrgHists(int ecalSubDet);
      void bookTimingHists(int ecalSubDet, int numBins);
      void bookTimingFedHists(int ecalSubDet);
      void bookTimingTrgHists(int ecalSubDet);

      // Analyzing Functions
      void analyzeSuperClusters(const reco::SuperClusterCollection *scCollection, 
	    const EcalRecHitCollection *recHitCollection, std::vector<bool>& l1Triggers);
      std::vector<bool> determineTriggers(const edm::Event&, const edm::EventSetup& eventSetup);
      bool isExclusiveTrigger(int l1Trigger, std::vector<bool>& l1Triggers);
      int getEcalSubDet(EcalRecHitCollection::const_iterator hitItr);
      std::pair<float,float> getPosition(EcalRecHitCollection::const_iterator hitItr);
      bool doTiming(EcalRecHitCollection::const_iterator hitItr);
      int getFedId(DetId detId);

      // Filling Functions
      void fillClusterHists(int ecalSubDet, reco::SuperClusterCollection::const_iterator scItr);
      void fillEnergyHists(reco::SuperClusterCollection::const_iterator scItr, 
	    		   EcalRecHitCollection::const_iterator seedItr, 
			   EcalRecHitCollection::const_iterator secondItr);
      void fillOccupancyHists(reco::SuperClusterCollection::const_iterator scItr,
	    		      EcalRecHitCollection::const_iterator seedItr);
      void fillOccupancyTrgHists(EcalRecHitCollection::const_iterator seedItr, std::vector<bool>& l1Triggers);
      void fillTimingHists(EcalRecHitCollection::const_iterator seedItr);
      void fillTimingFedHists(EcalRecHitCollection::const_iterator seedItr);
      void fillTimingTrgHists(EcalRecHitCollection::const_iterator seedItr, std::vector<bool>& l1Triggers);
      void makeDirectory(int ecalSubDet);

      // Conversion Functions
      std::string intToString(int num);
      std::string ecalSubDetToString(int ecalSubDet);
      std::string l1TriggerToString(int l1Trigger);

      // ----------member data ---------------------------
      enum EcalSubDet {
	 EEM	= 0,
	 EB	= 1,
	 EEP 	= 2
      };

      enum L1Triggers {
	 DT	= 0,
	 CSC	= 1,
	 RPC	= 2,
	 HCAL	= 3,
	 ECAL	= 4
      };

      edm::InputTag ecalRecHitCollectionEB_;
      edm::InputTag ecalRecHitCollectionEE_;
      edm::InputTag superClusterCollectionEB_;
      edm::InputTag superClusterCollectionEE_;
      edm::InputTag l1GTReadoutRecTag_;
      edm::InputTag l1GMTReadoutRecTag_;

      double histRangeMax_, histRangeMin_;
      double minTimingAmpEB_;
      double minTimingAmpEE_;
      double minRecHitAmpEB_;
      double minRecHitAmpEE_;
      double minHighEnergy_;

      DQMStore *dbe_;
      bool saveFile_;
      std::string fileName_;

      // Cluster Hists
      MonitorElement* h1f_NumXtalsInCluster_[3];
      MonitorElement* h2f_NumXtalsVsEnergy_[3];
      MonitorElement* h1f_NumBCinSC_[3];
      MonitorElement* h2f_NumBCinSCphi_;

      // Energy Hists
      MonitorElement* h1f_SeedEnergy_[3];
      MonitorElement* h1f_E2_[3];
      MonitorElement* h2f_energyvsE1_[3];
      MonitorElement* h1f_energy_[3];
      MonitorElement* h1f_energyHigh_[3];
      MonitorElement* h1f_energy_SingleXtalClusters_[3];

      // Occupancy Hists
      MonitorElement* h2f_Occupancy_[3];
      MonitorElement* h2f_OccupancyCoarse_[3];
      MonitorElement* h2f_OccupancyHighEnergyEvents_[3];
      MonitorElement* h2f_OccupancyHighEnergyEventsCoarse_[3];
      MonitorElement* h2f_OccupancySingleXtal_[3];

      // Specific Triggered Occupancy Hists
      MonitorElement* h2f_Occupancy_Trg_[3][5];
      MonitorElement* h2f_OccupancyCoarse_Trg_[3][5];
      MonitorElement* h2f_Occupancy_Exclusive_Trg_[3][5];
      MonitorElement* h2f_OccupancyCoarse_Exclusive_Trg_[3][5];

      // Timing Hists
      MonitorElement* h1f_timing_[3];
      MonitorElement* h3f_timingMod_;
      MonitorElement* h3f_timingTT_[3];
      MonitorElement* h2f_timingVsPhi_;
      MonitorElement* h2f_timingVsAmp_[3];
      std::map<int,MonitorElement*> h1f_timingFED_;
      MonitorElement* h1f_timingEBM_;
      MonitorElement* h1f_timingEBMTop_;
      MonitorElement* h1f_timingEBMBottom_;
      MonitorElement* h1f_timingEBP_;
      MonitorElement* h1f_timingEBPTop_;
      MonitorElement* h1f_timingEBPBottom_;

      // Specific Triggered Timing Hists
      MonitorElement* h3f_timingMod_Trg_[5];
      MonitorElement* h3f_timingTT_Trg_[3][5];
      MonitorElement* h1f_timing_Trg_[3][5];

      // Miscelanious?
      MonitorElement* h1f_triggerHist_[3];
      MonitorElement* h1f_triggerExclusiveHist_[3];
//      MonitorElement* h1f_FrequencyInTime_; // but need to figure out binning
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

#endif

