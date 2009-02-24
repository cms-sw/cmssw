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
// $Id: EcalOfflineCosmicTask.cc,v 1.1 2009/02/01 19:38:46 slynch Exp $
//
//

#include "DQMOffline/Ecal/interface/EcalOfflineCosmicTask.h"

#include "DataFormats/DetId/interface/DetId.h"
//#include "Geometry/CaloTopology/interface/CaloTopology.h"
//#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
//#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
//#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

//#include "DataFormats/EgammaReco/interface/BasicCluster.h"
//#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
//#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

//#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
//#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

//#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

using namespace std;

//
// constructors and destructor
//
EcalOfflineCosmicTask::EcalOfflineCosmicTask(const edm::ParameterSet& iConfig) :
   ecalRecHitCollectionEB_ (iConfig.getParameter<edm::InputTag>("ecalRecHitCollectionEB")),
   ecalRecHitCollectionEE_ (iConfig.getParameter<edm::InputTag>("ecalRecHitCollectionEE")),
   superClusterCollectionEB_ (iConfig.getParameter<edm::InputTag>("superClusterCollectionEB")),
   superClusterCollectionEE_ (iConfig.getParameter<edm::InputTag>("superClusterCollectionEE")),
   l1GTReadoutRecTag_ (iConfig.getUntrackedParameter<std::string>("L1GlobalReadoutRecord","gtDigis")),
   l1GMTReadoutRecTag_ (iConfig.getUntrackedParameter<std::string>("L1GlobalMuonReadoutRecord","gtDigis")),
//   runNum_(-1),
   histRangeMax_ (iConfig.getUntrackedParameter<double>("histogramMaxRange",1.8)),
   histRangeMin_ (iConfig.getUntrackedParameter<double>("histogramMinRange",0.0)),
   minTimingAmpEB_ (iConfig.getUntrackedParameter<double>("MinTimingAmpEB",0.100)),
   minTimingAmpEE_ (iConfig.getUntrackedParameter<double>("MinTimingAmpEE",0.100)),
   minRecHitAmpEB_(iConfig.getUntrackedParameter<double>("MinRecHitAmpEB",0.027)),
   minRecHitAmpEE_(iConfig.getUntrackedParameter<double>("MinRecHitAmpEE",0.18)),
   minHighEnergy_(iConfig.getUntrackedParameter<double>("MinHighEnergy",2.0)),
   saveFile_ (iConfig.getUntrackedParameter<bool>("saveFile",false)),
   fileName_ (iConfig.getUntrackedParameter<std::string>("fileName", std::string("ecalOfflineCosmicTask.root")))
{
   //now do what ever initialization is needed

}


EcalOfflineCosmicTask::~EcalOfflineCosmicTask()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalOfflineCosmicTask::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

     std::vector<bool> l1Triggers = determineTriggers(iEvent, iSetup);
    
     edm::Handle<reco::SuperClusterCollection> scHandleEB;
     iEvent.getByLabel(superClusterCollectionEB_, scHandleEB);
     if (!(scHandleEB.isValid())) {
	LogWarning("EcalOfflineCosmicTask") << superClusterCollectionEB_ << " not available";
	return;
     }

     Handle<EcalRecHitCollection> recHitsEB;
     iEvent.getByLabel(ecalRecHitCollectionEB_, recHitsEB);
     if (!(recHitsEB.isValid())) {
	LogWarning("EcalOfflineCosmicTask") << ecalRecHitCollectionEB_ << " not available";
	return; 
     }  

     const EcalRecHitCollection *recHitCollectionEB = recHitsEB.product();
     const reco::SuperClusterCollection *scCollectionEB = scHandleEB.product();
     analyzeSuperClusters(scCollectionEB,recHitCollectionEB,l1Triggers);

     edm::Handle<reco::SuperClusterCollection> scHandleEE;
     iEvent.getByLabel(superClusterCollectionEE_, scHandleEE);
     if (!(scHandleEE.isValid())) {
	LogWarning("EcalOfflineCosmicTask") << superClusterCollectionEE_ << " not available";
	//return;
     }

     Handle<EcalRecHitCollection> recHitsEE; 
     iEvent.getByLabel(ecalRecHitCollectionEE_, recHitsEE);
     if (!(recHitsEE.isValid())) {
	LogWarning("EcalOfflineCosmicTask") << ecalRecHitCollectionEE_ << " not available";
	return; 
     }

     const EcalRecHitCollection *recHitCollectionEE = recHitsEE.product();
     const reco::SuperClusterCollection *scCollectionEE = scHandleEE.product();
     analyzeSuperClusters(scCollectionEE,recHitCollectionEE,l1Triggers);
}


// ------------ method called once each job just before starting event loop  ------------


void 
EcalOfflineCosmicTask::beginJob(const edm::EventSetup& iSetup)
{
   Numbers::initGeometry(iSetup);
   dbe_ = edm::Service<DQMStore>().operator->();
   unsigned int numBins = 200;
   for(int ecalSubDet=0; ecalSubDet!=3; ++ecalSubDet) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
      // Book Cluster Hists
      bookClusterHists(ecalSubDet,numBins);

      // Book Energy Hists
      bookEnergyHists(ecalSubDet,numBins);

      // Book Occupancy Hists
      bookOccupancyHists(ecalSubDet);
      bookOccupancyTrgHists(ecalSubDet);

      // Book Timing Hists
      bookTimingHists(ecalSubDet,numBins);
      bookTimingTrgHists(ecalSubDet);

      // Miscellanious
      std::string name="triggerHist"+ecalSubDetString;
      std::string title="Trigger Number "+ecalSubDetString;
      h1f_triggerHist_[ecalSubDet] = dbe_->book1D(name,title,5,0,5);

      name="triggerExclusiveHist"+ecalSubDetString;
      title="Trigger Number (Mutually Exclusive) "+ecalSubDetString;
      h1f_triggerExclusiveHist_[ecalSubDet] = dbe_->book1D(name,title,5,0,5);
   }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalOfflineCosmicTask::endJob() {
   if(saveFile_)
      dbe_->save(fileName_);
}

///******************************** Begin Booking Functions *************************************************///
std::vector<float> EcalOfflineCosmicTask::makeXBins(int ecalSubDet, bool isCoarse) {
   std::vector<float> xBins;
   if(ecalSubDet == EB) {
      if(isCoarse) {
	 for(int i=0;i!=73;++i) 
	    xBins.push_back(1+i*5);
      }
      else {
	 for(int i=0;i!=361;++i)
	    xBins.push_back(i+1);
      }
   }
   else {
      if(isCoarse) {
	 for(int i=0;i!=21;++i)
	    xBins.push_back(i*5+1);
      }
      else {
	 for(int i=0;i!=101;++i)
	    xBins.push_back(i+1);
      }
   }
   return xBins;
}

std::vector<float> EcalOfflineCosmicTask::makeYBins(int ecalSubDet, bool isCoarse) {
   std::vector<float> yBins;
   if(ecalSubDet == EB) {
      if(isCoarse) {
	 for(int i=-85; i!=87;++i) {
	    if(i < 0 && i % 5 == 0)
	       yBins.push_back(i);
	    if(i == 0)
	       yBins.push_back(i);
	    if(i > 0 && (i-1) % 5 == 0)
	       yBins.push_back(i);
	 }
//	 yBins = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30,
//	    -25, -20, -15, -10,  -5,   0,   1,   6,  11,  16,  21,  26, 
//	    31,  36,  41,  46,  51,  56,  61,  66,  71,  76,  81,  86 };
      }
      else {
	 for(int i=0;i!=173;++i)
	    yBins.push_back(i-86);
      }
   }
   else {
      if(isCoarse) {
	 for(int i=0;i!=21;++i)
	    yBins.push_back(i*5+1);
      }
      else {
	 for(int i=0;i!=101;++i)
	    yBins.push_back(i+1);
      }
   }
   return yBins;
}

int EcalOfflineCosmicTask::getNumBins(std::vector<float> bins) {
   return bins.size() - 1;
}

void EcalOfflineCosmicTask::bookClusterHists(int ecalSubDet, int numBins) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/AllEvents/ClusterHists");
   // Book Cluster Hists
   std::string name = "NumXtalsInCluster";
   std::string title = "Number of Xtals in Cluster "+ecalSubDetString+";NumXtals";
   h1f_NumXtalsInCluster_[ecalSubDet] = dbe_->book1D(name,title,150,0,150);

   name = "NumXtalsVsEnergy2D";
   title = "Number of Xtals in Cluster vs. Energy "+ecalSubDetString+";Energy (GeV);Number of Xtals in Cluster";
   h2f_NumXtalsVsEnergy_[ecalSubDet] = dbe_->book2D(name,title,numBins,histRangeMin_,10.0,150,0,150);

   name = "NumBCinSC";
   title = "Number of Basic Clusters in Super Cluster "+ecalSubDetString+";Num Basic Clusters";
   h1f_NumBCinSC_[ecalSubDet] = dbe_->book1D(name,title,20,0,20);

   if(ecalSubDet == EB) {
      name = "NumBCinSCphi2D";
      title = "Number of Basic Clusters in Super Cluster "+ecalSubDetString+";#phi,Num Basic Clusters";
      h2f_NumBCinSCphi_ = dbe_->book2D(name,title,360/5,-3.14159,3.14159,20,0,20);
   }
}

void EcalOfflineCosmicTask::bookEnergyHists(int ecalSubDet, int numBins) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/AllEvents/EnergyHists");
   // Book Energy Hists
   std::string name = "E1";
   std::string title = "Seed Energy for All Feds "+ecalSubDetString+";Seed Energy (GeV)";
   h1f_SeedEnergy_[ecalSubDet] = dbe_->book1D(name,title,numBins,histRangeMin_,histRangeMax_);

   name = "E2";
   title = "E2 "+ecalSubDetString+";Seed+highestneighbor energy (GeV)";
   h1f_E2_[ecalSubDet] = dbe_->book1D(name,title,numBins,histRangeMin_,histRangeMax_);

   name = "EnergyVsE1";
   title = "Energy vs. E1 All Clusters "+ecalSubDetString+";Seed Energy (GeV);Energy(GeV)";
   h2f_energyvsE1_[ecalSubDet] = 
      dbe_->book2D(name,title,numBins,histRangeMin_,histRangeMax_,numBins,histRangeMin_,histRangeMax_);

   name = "Energy";
   title = "Energy All Clusters "+ecalSubDetString+";Cluster Energy (GeV)";
   h1f_energy_[ecalSubDet] = dbe_->book1D(name,title,numBins,histRangeMin_,histRangeMax_);

   name = "EnergyHigh";
   title = "Energy High All Clusters "+ecalSubDetString+";Cluster Energy (GeV)";
   h1f_energyHigh_[ecalSubDet] = dbe_->book1D(name,title,numBins,histRangeMin_,200.0);

   name = "Energy_SingleXtalClusters";
   title = "Energy single xtal clusters "+ecalSubDetString+";Cluster Energy (GeV)";
   h1f_energy_SingleXtalClusters_[ecalSubDet] = dbe_->book1D(name,title,numBins,histRangeMin_,200.0);
}

void EcalOfflineCosmicTask::bookOccupancyHists(int ecalSubDet) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/AllEvents/OccupancyHists");
   std::string xTitle = (ecalSubDet == EB) ? "i#phi" : "ix";
   std::string yTitle = (ecalSubDet == EB) ? "i#eta" : "iy";

   std::vector<float> xBins = makeXBins(ecalSubDet,false);
   int numXBins = getNumBins(xBins);
   std::vector<float> yBins = makeYBins(ecalSubDet,false);
   int numYBins = getNumBins(yBins);

   std::vector<float> ttXBins = makeXBins(ecalSubDet,true);
   int numTTXBins = getNumBins(ttXBins);
   std::vector<float> ttYBins = makeYBins(ecalSubDet,true);
   int numTTYBins = getNumBins(ttYBins);

   // Book Occupancy Hists
   std::string name="Occupancy";
   std::string title="Occupancy "+ecalSubDetString+";"+xTitle+";"+yTitle;
   h2f_Occupancy_[ecalSubDet] = 
      dbe_->book2D(name,title,numXBins,&xBins[0],numYBins,&yBins[0]);

   name="OccupancyCoarse";
   title="Occupancy Coarse "+ecalSubDetString+";"+xTitle+";"+yTitle;
   h2f_OccupancyCoarse_[ecalSubDet] = 
      dbe_->book2D(name,title,numTTXBins,&ttXBins[0],numTTYBins,&ttYBins[0]);

   name="OccupancyHighEnergyEvents";
   title="Occupancy high energy events "+ecalSubDetString+";"+xTitle+";"+yTitle;
   h2f_OccupancyHighEnergyEvents_[ecalSubDet] = 
      dbe_->book2D(name,title,numXBins,&xBins[0],numYBins,&yBins[0]);

   name="OccupancyHighEnergyEventsCoarse";
   title="Occupancy high energy events Coarse "+ecalSubDetString+";"+xTitle+";"+yTitle;
   h2f_OccupancyHighEnergyEventsCoarse_[ecalSubDet] = 
      dbe_->book2D(name,title,numTTXBins,&ttXBins[0],numTTYBins,&ttYBins[0]);

   name="OccupancySingleXtal";
   title="Occupancy single xtal clusters "+ecalSubDetString+";"+xTitle+";"+yTitle;
   h2f_OccupancySingleXtal_[ecalSubDet] = 
      dbe_->book2D(name,title,numXBins,&xBins[0],numYBins,&yBins[0]);
}

void EcalOfflineCosmicTask::bookOccupancyTrgHists(int ecalSubDet) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   std::string xTitle = (ecalSubDet == EB) ? "i#phi" : "ix";
   std::string yTitle = (ecalSubDet == EB) ? "i#eta" : "iy";

   std::vector<float> xBins = makeXBins(ecalSubDet,false);
   int numXBins = getNumBins(xBins);
   std::vector<float> yBins = makeYBins(ecalSubDet,false);
   int numYBins = getNumBins(yBins);

   std::vector<float> ttXBins = makeXBins(ecalSubDet,true);
   int numTTXBins = getNumBins(ttXBins);
   std::vector<float> ttYBins = makeYBins(ecalSubDet,true);
   int numTTYBins = getNumBins(ttYBins);

   for(int l1Trigger = 0; l1Trigger != 5; ++l1Trigger) {
      std::string l1TriggerString = l1TriggerToString(l1Trigger);
      dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/"+l1TriggerString+"/OccupancyHists");

      std::string name="Occupancy";
      std::string title="Occupancy "+ecalSubDetString+" "+l1TriggerString+" triggered;"
	 +xTitle+";"+yTitle;
      h2f_Occupancy_Trg_[ecalSubDet][l1Trigger] = 
	 dbe_->book2D(name,title,numXBins,&xBins[0],numYBins,&yBins[0]);

      name="OccupancyCoarse";
      title="Occupancy Coarse "+ecalSubDetString+" "+l1TriggerString+" triggered;"
	 +xTitle+";"+yTitle;
      h2f_OccupancyCoarse_Trg_[ecalSubDet][l1Trigger] = 
	 dbe_->book2D(name,title,numTTXBins,&ttXBins[0],numTTYBins,&ttYBins[0]);

      name="Occupancy_ExclusiveTrigger";
      title="Occupancy "+ecalSubDetString+" "+l1TriggerString+" triggered;"
	 +xTitle+";"+yTitle;
      h2f_Occupancy_Exclusive_Trg_[ecalSubDet][l1Trigger] = 
	 dbe_->book2D(name,title,numXBins,&xBins[0],numYBins,&yBins[0]);

      name="OccupancyCoarse_ExclusiveTrigger";
      title="Occupancy Coarse "+ecalSubDetString+" "+l1TriggerString+" triggered;"
	 +xTitle+";"+yTitle;
      h2f_OccupancyCoarse_Exclusive_Trg_[ecalSubDet][l1Trigger] = 
	 dbe_->book2D(name,title,numTTXBins,&ttXBins[0],numTTYBins,&ttYBins[0]);

   }
}

void EcalOfflineCosmicTask::bookTimingHists(int ecalSubDet, int numBins) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/AllEvents/TimingHists");
   std::string xTitle = (ecalSubDet == EB) ? "i#phi" : "ix";
   std::string yTitle = (ecalSubDet == EB) ? "i#eta" : "iy";

   std::vector<float> ttXBins = makeXBins(ecalSubDet,true);
   int numTTXBins = getNumBins(ttXBins);
   std::vector<float> ttYBins = makeYBins(ecalSubDet,true);
   int numTTYBins = getNumBins(ttYBins);

   // Book Timing Histograms
   std::string name ="timeForAllFeds";
   std:: string title="Time for all feds "+ecalSubDetString+";Relative Time (1 clock = 25ns)";
   h1f_timing_[ecalSubDet] = dbe_->book1D(name,title,78,-7,7);

   float modEtaBins[10]={-85, -65, -45, -25, 0, 1, 26, 46, 66, 86};
   float modPhiBins[19];
   float timingBins[79];
   for (int i = 0; i < 79; ++i) {
      timingBins[i]=-7.+float(i)*14./78.;
      if ( i < 19) {       
	 modPhiBins[i]=1+20*i;
      }       
   }

   if(ecalSubDet == EB) {
      name = "timingPhi";
      title = "Time vs phi for all FEDs "+ecalSubDetString+";i#phi;RelativeTime (1 clock = 25ns)";
      h2f_timingVsPhi_ = dbe_->book2D(name,title,72,1,361,78,-7,7);

      name = "timingModBinning3D";
      title = "(i#phi,i#eta,time) (Module Binning);i#phi,i#eta,Relative Time (1 clock = 25ns)";
   // No constructor for a 3D MonitorElement with Variable Bins, very annoying
      TH3F* tmp3f_timePhiEta = 
	 new TH3F(name.c_str(),title.c_str(),18,&modPhiBins[0],9,&modEtaBins[0],78,&timingBins[0]);
      h3f_timingMod_ = dbe_->book3D(name,tmp3f_timePhiEta);
      delete tmp3f_timePhiEta;

   }

   name = "timingVsAmp2D";
   title = "Time Vs. Amp "+ecalSubDetString+";Relative Time (1 clock = 25ns);Amplitude (GeV)";
   h2f_timingVsAmp_[ecalSubDet] = dbe_->book2D(name,title,78,-7,7,numBins,histRangeMin_,histRangeMax_);

   name = "timingTTBinning3D";
   title = 
      "("+xTitle+","+yTitle+",time) "+ecalSubDetString+";"+xTitle+";"+yTitle+";RelativeTime (1 clock = 25ns)";
   // No constructor for a 3D MonitorElement with Variable Bins, very annoying
   TH3F* tmp3f_timeTT = 
      new TH3F(name.c_str(),title.c_str(),numTTXBins,&ttXBins[0],numTTYBins,&ttYBins[0],78,timingBins);
   h3f_timingTT_[ecalSubDet] = dbe_->book3D(name,tmp3f_timeTT);
   delete tmp3f_timeTT;

   bookTimingFedHists(ecalSubDet);
}

void EcalOfflineCosmicTask::bookTimingFedHists(int ecalSubDet) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/AllEvents/FEDTimingHists");
   int startFed = 0;
   int stopFed = 0;
   if(ecalSubDet == EB)  {
      startFed = 610;
      stopFed = 645;
   }
   else if(ecalSubDet == EEM) {
      startFed = 601;
      stopFed = 609;
   }
   else if(ecalSubDet == EEP) {
      startFed = 646;
      stopFed = 654;
   }
   for(int fedId=startFed; fedId!=stopFed+1; ++fedId) {
      std::string name="TimingFed"+intToString(fedId);
      std::string title="Timing for FED "+intToString(fedId)+";Relative Time (1 clock = 25ns)";
      h1f_timingFED_[fedId] = dbe_->book1D(name,title,78,-7,7);
   }
   std::string name = "timingEBM";
   std::string title = "Time For FEDs in EB-;Relative Time (1 clock = 25ns)";
   h1f_timingEBM_ = dbe_->book1D(name,title,78,-7,7);

   name = "timingEBMTop";
   title = "Time For FEDs in EB- Top;Relative Time (1 clock = 25ns)";
   h1f_timingEBMTop_ = dbe_->book1D(name,title,78,-7,7);

   name = "timingEBMBottom";
   title = "Time For FEDs in EB- Bottom;Relative Time (1 clock = 25ns)";
   h1f_timingEBMBottom_ = dbe_->book1D(name,title,78,-7,7);

   name = "timingEBP";
   title = "Time For FEDs in EB+;Relative Time (1 clock = 25ns)";
   h1f_timingEBP_ = dbe_->book1D(name,title,78,-7,7);

   name = "timingEBPTop";
   title = "Time For FEDs in EB+ Top;Relative Time (1 clock = 25ns)";
   h1f_timingEBPTop_ = dbe_->book1D(name,title,78,-7,7);

   name = "timingEBPBottom";
   title = "Time For FEDs in EB+ Bottom;Relative Time (1 clock = 25ns)";
   h1f_timingEBPBottom_ = dbe_->book1D(name,title,78,-7,7);
}

void EcalOfflineCosmicTask::bookTimingTrgHists(int ecalSubDet) {
   std::string ecalSubDetString = ecalSubDetToString(ecalSubDet);
   std::string xTitle = (ecalSubDet == EB) ? "i#phi" : "ix";
   std::string yTitle = (ecalSubDet == EB) ? "i#eta" : "iy";

   std::vector<float> ttXBins = makeXBins(ecalSubDet,true);
   int numTTXBins = getNumBins(ttXBins);
   std::vector<float> ttYBins = makeYBins(ecalSubDet,true);
   int numTTYBins = getNumBins(ttYBins);

   float modEtaBins[10]={-85, -65, -45, -25, 0, 1, 26, 46, 66, 86};
   float modPhiBins[19];
   float timingBins[79];
   for (int i = 0; i < 79; ++i) {
      timingBins[i]=-7.+float(i)*14./78.;
      if ( i < 19) {       
	 modPhiBins[i]=1+20*i;
      }       
   }

   for(int l1Trigger = 0; l1Trigger != 5; ++l1Trigger) {
      std::string l1TriggerString = l1TriggerToString(l1Trigger);
   dbe_->setCurrentFolder("EcalOfflineCosmicTask/"+ecalSubDetString+"/"+l1TriggerString+"/TimingHists");

      if(ecalSubDet == EB) {
	 std::string name = "timingModBinning3D";
	 std::string title = "(i#phi,i#eta,time) (Module Binning) "+l1TriggerString;
	 TH3F* tmp3f_timePhiEta_Trg = 
	    new TH3F(name.c_str(),title.c_str(),18,&modPhiBins[0],9,&modEtaBins[0],78,&timingBins[0]);
	 h3f_timingMod_Trg_[l1Trigger] = dbe_->book3D(name,tmp3f_timePhiEta_Trg);
	 delete tmp3f_timePhiEta_Trg;
      }

      std::string name = "timingTTBinning3D";
      std::string title = "("+xTitle+","+yTitle+",time) (TT Binning) "+l1TriggerString;
      TH3F* tmp3f_timeTT_Trg =
	 new TH3F(name.c_str(),title.c_str(),numTTXBins,&ttXBins[0],numTTYBins,&ttYBins[0],78,timingBins);
      h3f_timingTT_Trg_[ecalSubDet][l1Trigger] = dbe_->book3D(name,tmp3f_timeTT_Trg);
      delete tmp3f_timeTT_Trg;

      name = "timeForAllFeds";
      title = "Time for all FEDs "+ ecalSubDetString+" "+l1TriggerString+";Relative Time(1 Clock = 25ns)";
      h1f_timing_Trg_[ecalSubDet][l1Trigger] = dbe_->book1D(name,title,78,-7,7);
   }
}
///******************************* End Booking Functions ****************************************************///

///***************************** Begin Analyzing Functinos **************************************************///
void EcalOfflineCosmicTask::analyzeSuperClusters(const reco::SuperClusterCollection *scCollection,
      const EcalRecHitCollection *recHitCollection, std::vector<bool>& l1Triggers) {
   using namespace edm;

   for(reco::SuperClusterCollection::const_iterator scItr = scCollection->begin();
	 scItr != scCollection->end(); ++scItr) {

     vector< pair<DetId,float> > scDetIds = scItr->hitsAndFractions();
      
      EcalRecHitCollection::const_iterator seedItr = recHitCollection->begin();
      EcalRecHitCollection::const_iterator secondItr = recHitCollection->begin();

      for(vector< pair<DetId,float> >::const_iterator detItr = scDetIds.begin(); detItr != scDetIds.end(); ++detItr) {
        DetId id = detItr->first;
	 if(id.det() != DetId::Ecal) { continue; }
	 EcalRecHitCollection::const_iterator hitItr = recHitCollection->find(id);
	 if(hitItr == recHitCollection->end()) { continue; }
	 if(hitItr->energy() > secondItr->energy()) { secondItr = hitItr; }
	 if(hitItr->energy() > seedItr->energy()) { std::swap(seedItr,secondItr); }
      }
      
      int ecalSubDet=getEcalSubDet(seedItr);
      if(ecalSubDet == -1) { continue; }

      // Cluster Hists
	fillClusterHists(ecalSubDet,scItr);
	
	// Energy Hists
	fillEnergyHists(scItr,seedItr,secondItr);

	// Occupancy Hists
	fillOccupancyHists(scItr,seedItr);
	fillOccupancyTrgHists(seedItr,l1Triggers);
	
	// Timing Hists
	if(doTiming(seedItr)) {
	   fillTimingHists(seedItr);
	   fillTimingTrgHists(seedItr,l1Triggers);
	}
     }

}

std::vector<bool> 
EcalOfflineCosmicTask::determineTriggers(const edm::Event& iEvent, const edm::EventSetup& eventSetup) {

   using namespace edm;
   std::vector<bool> l1Triggers; //DT,CSC,RPC,HCAL,ECAL
   //0 , 1 , 2 , 3  , 4
   for(int i=0;i<5;i++)
      l1Triggers.push_back(false);

   // get the GMTReadoutCollection
   edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
   iEvent.getByLabel(l1GMTReadoutRecTag_,gmtrc_handle);
   L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
   if (!(gmtrc_handle.isValid()))  
   {
      LogWarning("EcalCosmicsHists") << "l1MuGMTReadoutCollection" << " not available";
      return l1Triggers;
   }  
   // get hold of L1GlobalReadoutRecord
   edm::Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
   iEvent.getByLabel(l1GTReadoutRecTag_,L1GTRR);

   //Ecal
   edm::ESHandle<L1GtTriggerMenu> menuRcd;
   eventSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
   const L1GtTriggerMenu* menu = menuRcd.product();
   edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
   iEvent.getByLabel( edm::InputTag("gtDigis"), gtRecord);
   // Get dWord after masking disabled bits
   const DecisionWord dWord = gtRecord->decisionWord();

   bool l1SingleEG2 = menu->gtAlgorithmResult("L1_SingleEG2", dWord);
   bool l1SingleEG5 = menu->gtAlgorithmResult("L1_SingleEG5", dWord);
   bool l1SingleEG8 = menu->gtAlgorithmResult("L1_SingleEG8", dWord);
   bool l1SingleEG10 = menu->gtAlgorithmResult("L1_SingleEG10", dWord);
   bool l1SingleEG12 = menu->gtAlgorithmResult("L1_SingleEG12", dWord);
   bool l1SingleEG15 = menu->gtAlgorithmResult("L1_SingleEG15", dWord);
   bool l1SingleEG20 = menu->gtAlgorithmResult("L1_SingleEG20", dWord);
   bool l1SingleEG25 = menu->gtAlgorithmResult("L1_SingleEG25", dWord);
   bool l1DoubleNoIsoEGBTBtight = menu->gtAlgorithmResult("L1_DoubleNoIsoEG_BTB_tight", dWord);
   bool l1DoubleNoIsoEGBTBloose = menu->gtAlgorithmResult("L1_DoubleNoIsoEG_BTB_loose ", dWord);
   bool l1DoubleNoIsoEGTopBottom = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottom", dWord);
   bool l1DoubleNoIsoEGTopBottomCen  = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCen", dWord);
   bool l1DoubleNoIsoEGTopBottomCen2  = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCen2", dWord);
   bool l1DoubleNoIsoEGTopBottomCenVert  = menu->gtAlgorithmResult("L1_DoubleNoIsoEGTopBottomCenVert", dWord);

   l1Triggers[ECAL] = l1SingleEG2 || l1SingleEG5 || l1SingleEG8 || l1SingleEG10 || l1SingleEG12 || l1SingleEG15
      || l1SingleEG20 || l1SingleEG25 || l1DoubleNoIsoEGBTBtight || l1DoubleNoIsoEGBTBloose
      || l1DoubleNoIsoEGTopBottom || l1DoubleNoIsoEGTopBottomCen || l1DoubleNoIsoEGTopBottomCen2
      || l1DoubleNoIsoEGTopBottomCenVert;

   std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
   std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
   for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
      std::vector<L1MuRegionalCand>::const_iterator iter1;
      std::vector<L1MuRegionalCand> rmc;

      //DT triggers
      int idt = 0;
      rmc = igmtrr->getDTBXCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	 if ( !(*iter1).empty() ) {
	    idt++;
	 }
      }
      //if(idt>0) std::cout << "Found " << idt << " valid DT candidates in bx wrt. L1A = " 
      //  << igmtrr->getBxInEvent() << std::endl;
      if(igmtrr->getBxInEvent()==0 && idt>0) l1Triggers[DT] = true;

      //RPC triggers
      int irpcb = 0;
      rmc = igmtrr->getBrlRPCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	 if ( !(*iter1).empty() ) {
	    irpcb++;
	 }
      }
      //if(irpcb>0) std::cout << "Found " << irpcb << " valid RPC candidates in bx wrt. L1A = " 
      //  << igmtrr->getBxInEvent() << std::endl;
      if(igmtrr->getBxInEvent()==0 && irpcb>0) l1Triggers[RPC] = true;

      //CSC Triggers
      int icsc = 0;
      rmc = igmtrr->getCSCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	 if ( !(*iter1).empty() ) {
	    icsc++;
	 }
      }
      //if(icsc>0) std::cout << "Found " << icsc << " valid CSC candidates in bx wrt. L1A = " 
      //  << igmtrr->getBxInEvent() << std::endl;
      if(igmtrr->getBxInEvent()==0 && icsc>0) l1Triggers[CSC] = true;
   }

   L1GlobalTriggerReadoutRecord const* gtrr = L1GTRR.product();

   for(int ibx=-1; ibx<=1; ibx++) {
      bool hcal_top = false;
      bool hcal_bot = false;
      const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d,ibx);
      std::vector<int> valid_phi; 
      if((psb.aData(4)&0x3f) >= 1) {valid_phi.push_back( (psb.aData(4)>>10)&0x1f ); }
      if((psb.bData(4)&0x3f) >= 1) {valid_phi.push_back( (psb.bData(4)>>10)&0x1f ); }
      if((psb.aData(5)&0x3f) >= 1) {valid_phi.push_back( (psb.aData(5)>>10)&0x1f ); }
      if((psb.bData(5)&0x3f) >= 1) {valid_phi.push_back( (psb.bData(5)>>10)&0x1f ); }
      std::vector<int>::const_iterator iphi;
      for(iphi=valid_phi.begin(); iphi!=valid_phi.end(); iphi++) {
	 //std::cout << "Found HCAL mip with phi=" << *iphi << " in bx wrt. L1A = " << ibx << std::endl;
	 if(*iphi<9) hcal_top=true;
	 if(*iphi>8) hcal_bot=true;
      }
      if(ibx==0 && hcal_top && hcal_bot) l1Triggers[HCAL]=true;
   }     
   return l1Triggers;
}

bool EcalOfflineCosmicTask::isExclusiveTrigger(int l1Trigger, std::vector<bool>& l1Triggers) {
   bool isExclusive = true;
   for(int otherL1Trigger = 0; otherL1Trigger != 5; ++otherL1Trigger) {
      if(l1Trigger != otherL1Trigger && otherL1Trigger == true) {
	 isExclusive = false;
      }
   }
   return isExclusive;
}

int EcalOfflineCosmicTask::getEcalSubDet(EcalRecHitCollection::const_iterator hitItr) {
   int ret = -1;
   DetId detId = hitItr->id();
   if(detId.det() == DetId::Ecal) {
      if(detId.subdetId() == EcalBarrel)
	 ret = EB;
      else
	 if(((EEDetId)detId).zside() < 0)
	    ret = EEM;
	 else
	    ret = EEP;
   }
   return ret;
}

std::pair<float,float> EcalOfflineCosmicTask::getPosition(EcalRecHitCollection::const_iterator hitItr) {
   int ecalSubDet = getEcalSubDet(hitItr);
   std::pair<float,float> seedPos;
   DetId detId = hitItr->id();
   if(ecalSubDet == EB) {
      EBDetId ecalDetId = detId;
      seedPos = std::pair<float,float>(ecalDetId.iphi(),ecalDetId.ieta());
   }
   else {
      EEDetId ecalDetId = detId;
      seedPos = std::pair<float,float>(ecalDetId.ix(),ecalDetId.iy());
   }
   return seedPos;
}

bool EcalOfflineCosmicTask::doTiming(EcalRecHitCollection::const_iterator hitItr) {
   bool ret = false;
   int ecalSubDet = getEcalSubDet(hitItr);
   if(ecalSubDet == EB)
      ret = (hitItr->energy() > minTimingAmpEB_);
   else
      ret = (hitItr->energy() > minTimingAmpEE_);
   return ret;
}

int EcalOfflineCosmicTask::getFedId(DetId detId) {
   if(detId.subdetId() == EcalBarrel) {
      return Numbers::iSM((EBDetId)detId) + 609;
   }
   else {
      int eeISM = Numbers::iSM((EEDetId)detId);
      if(eeISM <= 9)
	 return eeISM + 600;
      else
	 return eeISM + 600 + 45 - 9;
   }
}
///******************************* End Analyzing Functinos **************************************************///

///****************************** Begin Filling Functions ***************************************************///
void EcalOfflineCosmicTask::fillClusterHists(int ecalSubDet, reco::SuperClusterCollection::const_iterator scItr)
{
   h1f_NumXtalsInCluster_[ecalSubDet]->Fill(scItr->size());
   h2f_NumXtalsVsEnergy_[ecalSubDet]->Fill(scItr->energy(),scItr->size());
   h1f_NumBCinSC_[ecalSubDet]->Fill(scItr->clustersSize());
   if(ecalSubDet == EB)
      h2f_NumBCinSCphi_->Fill(scItr->phi(),scItr->clustersSize());
}

void EcalOfflineCosmicTask::fillEnergyHists(reco::SuperClusterCollection::const_iterator scItr, 
      					    EcalRecHitCollection::const_iterator seedItr,
					    EcalRecHitCollection::const_iterator secondItr)
{
   int ecalSubDet = getEcalSubDet(seedItr);
   float scE1 = seedItr->energy();
   float scE2 = seedItr->energy() + secondItr->energy();
   h1f_SeedEnergy_[ecalSubDet]->Fill(scE1);
   h1f_E2_[ecalSubDet]->Fill(scE2);
   h2f_energyvsE1_[ecalSubDet]->Fill(scE1,scItr->energy());
   h1f_energy_[ecalSubDet]->Fill(scItr->energy());
   h1f_energyHigh_[ecalSubDet]->Fill(scItr->energy());
   if(scItr->size() == 1)
      h1f_energy_SingleXtalClusters_[ecalSubDet]->Fill(scItr->energy());
}

void 
EcalOfflineCosmicTask::fillOccupancyHists(reco::SuperClusterCollection::const_iterator scItr,
      					  EcalRecHitCollection::const_iterator seedItr) 
{
   int ecalSubDet = getEcalSubDet(seedItr);
   std::pair<float,float> seedPos = getPosition(seedItr);
   // Fill Hists
   h2f_Occupancy_[ecalSubDet]->Fill(seedPos.first,seedPos.second);
   h2f_OccupancyCoarse_[ecalSubDet]->Fill(seedPos.first,seedPos.second);
   if(scItr->energy() > minHighEnergy_) {
      h2f_OccupancyHighEnergyEvents_[ecalSubDet]->Fill(seedPos.first,seedPos.second);
      h2f_OccupancyHighEnergyEventsCoarse_[ecalSubDet]->Fill(seedPos.first,seedPos.second);
   }
   if(scItr->size() == 1) {
      h2f_OccupancySingleXtal_[ecalSubDet]->Fill(seedPos.first,seedPos.second);
   }


}

void EcalOfflineCosmicTask::fillOccupancyTrgHists(EcalRecHitCollection::const_iterator seedItr, 
      						  std::vector<bool>& l1Triggers) {
   int ecalSubDet = getEcalSubDet(seedItr);
   std::pair<float,float> seedPos = getPosition(seedItr);

   for(int l1Trigger = 0; l1Trigger != 5; ++l1Trigger) {
      if(l1Triggers[l1Trigger] == true) {
	 h1f_triggerHist_[ecalSubDet]->Fill(l1Trigger);
	 h2f_Occupancy_Trg_[ecalSubDet][l1Trigger]->Fill(seedPos.first,seedPos.second);
	 h2f_OccupancyCoarse_Trg_[ecalSubDet][l1Trigger]->Fill(seedPos.first,seedPos.second);
	 if(isExclusiveTrigger(l1Trigger,l1Triggers)) {
	    h1f_triggerExclusiveHist_[ecalSubDet]->Fill(l1Trigger);
	    h2f_Occupancy_Exclusive_Trg_[ecalSubDet][l1Trigger]->Fill(seedPos.first,seedPos.second);
	    h2f_OccupancyCoarse_Exclusive_Trg_[ecalSubDet][l1Trigger]->Fill(seedPos.first,seedPos.second);
	 }
      }
   }
}

void EcalOfflineCosmicTask::fillTimingHists(EcalRecHitCollection::const_iterator seedItr) 
{
   int ecalSubDet = getEcalSubDet(seedItr);
   std::pair<float,float> seedPos = getPosition(seedItr);

   h1f_timing_[ecalSubDet]->Fill(seedItr->time());
   if(ecalSubDet == EB) {
      h2f_timingVsPhi_->Fill(seedPos.first,seedItr->time());
      h3f_timingMod_->Fill(seedPos.first,seedPos.second,seedItr->time());
   }

   h3f_timingTT_[ecalSubDet]->Fill(seedPos.first,seedPos.second,seedItr->time());
   h2f_timingVsAmp_[ecalSubDet]->Fill(seedItr->time(),seedItr->energy());
   fillTimingFedHists(seedItr);
}

void EcalOfflineCosmicTask::fillTimingFedHists(EcalRecHitCollection::const_iterator seedItr) {
   int ecalSubDet = getEcalSubDet(seedItr);
   int fedId = getFedId(seedItr->id());

   h1f_timingFED_[fedId]->Fill(seedItr->time());

   if(ecalSubDet == EB) {
      if (fedId>=610&&fedId<=627) h1f_timingEBM_->Fill(seedItr->time());
      if (fedId>=628&&fedId<=645) h1f_timingEBP_->Fill(seedItr->time());
      if (fedId>=613&&fedId<=616) h1f_timingEBMTop_->Fill(seedItr->time());
      if (fedId>=631&&fedId<=634) h1f_timingEBPTop_->Fill(seedItr->time());
      if (fedId>=622&&fedId<=625) h1f_timingEBMBottom_->Fill(seedItr->time());
      if (fedId>=640&&fedId<=643) h1f_timingEBPBottom_->Fill(seedItr->time());
   }
}

void EcalOfflineCosmicTask::fillTimingTrgHists(EcalRecHitCollection::const_iterator seedItr,
      					       std::vector<bool>& l1Triggers) {
   int ecalSubDet = getEcalSubDet(seedItr);
   std::pair<float,float> seedPos = getPosition(seedItr);

   for(int l1Trigger = 0; l1Trigger != 5; ++l1Trigger) {
      if(l1Triggers[l1Trigger] == true) {
	 if(ecalSubDet == EB)
	    h3f_timingMod_Trg_[l1Trigger]->Fill(seedPos.first,seedPos.second,seedItr->time());
	 h3f_timingTT_Trg_[ecalSubDet][l1Trigger]->Fill(seedPos.first,seedPos.second,seedItr->time());
	 h1f_timing_Trg_[ecalSubDet][l1Trigger]->Fill(seedItr->time());
      }
   }
}
///******************************** End Filling Functions ***************************************************///

///****************************** Begin Conversion Functions ************************************************///
std::string EcalOfflineCosmicTask::intToString(int num) {
   using namespace std;
   ostringstream myStream;
   myStream << num << flush;
   return(myStream.str()); //returns the string form of the stringstream object
}

std::string EcalOfflineCosmicTask::ecalSubDetToString(int ecalSubDet) {
   std::string ret="";
   switch(ecalSubDet) {
      case 0:  ret="EEM"; break;
      case 1:  ret="EB";  break;
      case 2:  ret="EEP"; break;
      default: ret="";    break;
   }
   return ret;
}

std::string EcalOfflineCosmicTask::l1TriggerToString(int l1Trigger) {
   std::string ret = "";
   switch(l1Trigger) {
      case 0:  ret = "DT";   break;
      case 1:  ret = "CSC";  break;
      case 2:  ret = "RPC";  break;
      case 3:  ret = "HCAL"; break;
      case 4:  ret = "ECAL"; break;
      default: ret = "";     break;
   }
   return ret;
}
///******************************** End Conversion Functions ************************************************///

//define this as a plug-in
