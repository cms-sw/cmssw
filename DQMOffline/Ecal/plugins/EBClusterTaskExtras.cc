/*
 * \file EBClusterTaskExtras.cc
 *
 * \author G. Della Ricca
 * \author E. Di Marco
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQMOffline/Ecal/interface/EBClusterTaskExtras.h>

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;


EBClusterTaskExtras::EBClusterTaskExtras(const ParameterSet& ps){

   init_ = false;

   dqmStore_ = Service<DQMStore>().operator->();

   prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

   enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

   mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

   // parameters...
   SuperClusterCollection_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("SuperClusterCollection"));
   EcalRecHitCollection_ = consumes<EcalRecHitCollection>(ps.getParameter<edm::InputTag>("EcalRecHitCollection"));
   l1GTReadoutRecToken_ = consumes<L1GlobalTriggerReadoutRecord>(ps.getParameter<edm::InputTag>("l1GlobalReadoutRecord"));
   l1GMTReadoutRecToken_ = consumes<L1MuGMTReadoutCollection>(ps.getParameter<edm::InputTag>("l1GlobalMuonReadoutRecord"));

   // histograms...
#ifndef EBCLUSTERTASKEXTRAS_DQMOFFLINE
   meSCSizCrystal_ = 0;
   meSCSizBC_ = 0;
   meSCSizPhi_ = 0;

   meSCSeedEne_ = 0;
   meSCEne2_ = 0;
   meSCEneLow_ = 0;
   meSCEneHigh_ = 0;
   meSCEneSingleCrystal_ = 0;

   meSCSeedMapOccSingleCrystal_ = 0;
   meSCSeedMapOccTT_ = 0;
   meSCSeedMapOccHighEne_ = 0;

   meSCSeedTime_ = 0;
   meSCSeedMapTimeTT_ = 0;
   meSCSeedMapTimeMod_ = 0;
   meSCSeedTimeVsPhi_ = 0;
   meSCSeedTimeVsAmp_ = 0;
   meSCSeedTimeEBM_ = 0;
   meSCSeedTimeEBP_ = 0;
   meSCSeedTimeEBMTop_ = 0;
   meSCSeedTimeEBPTop_ = 0;
   meSCSeedTimeEBMBot_ = 0;
   meSCSeedTimeEBPBot_ = 0;
   for(int i=0;i!=36;++i)
      meSCSeedTimePerFed_[i] = 0;
   for(int i=0;i!=5;++i) {
      meSCSeedMapOccTrg_[i] = 0;
      meSCSeedMapOccTrgExcl_[i] = 0;
      meSCSeedMapTimeTrgMod_[i] = 0;
   }
#endif

   meSCSizCrystalVsEne_ = 0;

   meSCSeedMapOcc_ = 0;
   meSCSeedMapOccHighEneTT_ = 0;
   for(int i=0;i!=5;++i) {
      meSCSeedMapOccTrgTT_[i] = 0;
      meSCSeedMapOccTrgExclTT_[i] = 0;

      meSCSeedMapTimeTrgTT_[i] = 0;
      meSCSeedTimeTrg_[i] = 0;
   }

   meTrg_ = 0;
   meTrgExcl_ = 0;

}

EBClusterTaskExtras::~EBClusterTaskExtras(){

}

void EBClusterTaskExtras::beginJob(){

   ievt_ = 0;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras");
      dqmStore_->rmdir(prefixME_ + "/EBClusterTaskExtras");
   }

}

void EBClusterTaskExtras::beginRun(const Run& r, const EventSetup& c) {

   Numbers::initGeometry(c, false);

   if ( ! mergeRuns_ ) this->reset();

}

void EBClusterTaskExtras::endRun(const Run& r, const EventSetup& c) {

}

void EBClusterTaskExtras::reset(void) {
#ifndef EBCLUSTERTASKEXTRAS_DQMOFFLINE
   if ( meSCSizCrystal_ ) meSCSizCrystal_->Reset();
   if ( meSCSizBC_ ) meSCSizBC_->Reset(); 
   if ( meSCSizPhi_ ) meSCSizPhi_->Reset();

   if ( meSCSeedEne_ ) meSCSeedEne_->Reset();
   if ( meSCEne2_ ) meSCEne2_->Reset();
   if ( meSCEneLow_ ) meSCEneLow_->Reset();
   if ( meSCEneHigh_ ) meSCEneHigh_->Reset();
   if ( meSCEneSingleCrystal_ ) meSCEneSingleCrystal_->Reset();

   if ( meSCSeedMapOccSingleCrystal_ ) meSCSeedMapOccSingleCrystal_->Reset();
   if ( meSCSeedMapOccTT_ ) meSCSeedMapOccTT_->Reset();
   if ( meSCSeedMapOccHighEne_ ) meSCSeedMapOccHighEne_->Reset();

   if ( meSCSeedTime_ ) meSCSeedTime_->Reset();
   if ( meSCSeedMapTimeTT_ ) meSCSeedMapTimeTT_->Reset();
   if ( meSCSeedTimeVsPhi_ ) meSCSeedTimeVsPhi_->Reset();
   if ( meSCSeedTimeVsAmp_ ) meSCSeedTimeVsAmp_->Reset();
   if ( meSCSeedTimeEBM_ ) meSCSeedTimeEBM_->Reset();
   if ( meSCSeedTimeEBP_ ) meSCSeedTimeEBP_->Reset();
   if ( meSCSeedTimeEBMTop_ ) meSCSeedTimeEBMTop_->Reset();
   if ( meSCSeedTimeEBPTop_ ) meSCSeedTimeEBPTop_->Reset();
   if ( meSCSeedTimeEBMBot_ ) meSCSeedTimeEBMBot_->Reset();
   if ( meSCSeedTimeEBPBot_ ) meSCSeedTimeEBPBot_->Reset();

   for(int i=0;i!=36; ++i) 
      if ( meSCSeedTimePerFed_[i] ) meSCSeedTimePerFed_[i]->Reset();

   for(int i=0;i!=5;++i) {
      if ( meSCSeedMapOccTrg_[i] ) meSCSeedMapOccTrg_[i]->Reset();
      if ( meSCSeedMapOccTrgExcl_[i] ) meSCSeedMapOccTrgExcl_[i]->Reset();
      if ( meSCSeedMapTimeTrgMod_[i] ) meSCSeedMapTimeTrgMod_[i]->Reset();
   }

   if ( meSCSeedMapTimeMod_ ) meSCSeedMapTimeMod_->Reset();
#endif

   if ( meSCSizCrystalVsEne_ ) meSCSizCrystalVsEne_->Reset();

   if ( meSCSeedMapOcc_ ) meSCSeedMapOcc_->Reset();
   if ( meSCSeedMapOccHighEneTT_ ) meSCSeedMapOccHighEneTT_->Reset();


   for(int i=0; i!=5; ++i) {
      if ( meSCSeedMapOccTrgTT_[i] ) meSCSeedMapOccTrgTT_[i]->Reset();
      if ( meSCSeedMapOccTrgExclTT_[i] ) meSCSeedMapOccTrgExclTT_[i]->Reset();

      if ( meSCSeedMapTimeTrgTT_[i] ) meSCSeedMapTimeTrgTT_[i]->Reset();
      if ( meSCSeedTimeTrg_[i] ) meSCSeedTimeTrg_[i]->Reset();
   }
   if ( meTrg_ ) meTrg_->Reset();
   if ( meTrgExcl_ ) meTrgExcl_->Reset();
}

void EBClusterTaskExtras::setup(void){

   init_ = true;

   std::string histo;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras");

#ifndef EBCLUSTERTASKEXTRAS_DQMOFFLINE
      // Cluster hists
      histo = "EBCLTE SC size (crystal)";
      meSCSizCrystal_ = dqmStore_->book1D(histo,histo,150,0,150);
      meSCSizCrystal_->setAxisTitle("super cluster size (crystal)", 1);

      histo = "EBCLTE SC size (basic clusters)";
      meSCSizBC_ = dqmStore_->book1D(histo,histo,20,0,20);
      meSCSizBC_->setAxisTitle("super cluster size (basic clusters)", 1);

      histo = "EBCLTE SC size in basic clusters vs phi";
      meSCSizPhi_ = dqmStore_->bookProfile(histo,histo,360/5,-3.14159,3.14159,20,0,20);
      meSCSizPhi_->setAxisTitle("phi", 1);
      meSCSizPhi_->setAxisTitle("super cluster size (basic clusters)", 2);

      histo = "EBCLTE SC seed crystal energy";
      meSCSeedEne_ = dqmStore_->book1D(histo,histo,200,0,1.8);
      meSCSeedEne_->setAxisTitle("seed crystal energy (GeV)", 1);

      histo = "EBCLTE SC seed e2";
      meSCEne2_ = dqmStore_->book1D(histo,histo,200,0,1.8);
      meSCEne2_->setAxisTitle("seed + highest neighbor crystal energy (GeV)", 1);

      histo = "EBCLTE SC energy low scale";
      meSCEneLow_ = dqmStore_->book1D(histo,histo,200,0,1.8);
      meSCEneLow_->setAxisTitle("energy (GeV)", 1);

      histo = "EBCLTE SC energy high scale";
      meSCEneHigh_ = dqmStore_->book1D(histo,histo,200,0,200);
      meSCEneHigh_->setAxisTitle("energy (GeV)", 1);

      histo = "EBCLTE SC single crystal energy (GeV)";
      meSCEneSingleCrystal_ = dqmStore_->book1D(histo,histo,200,0,200);
      meSCEneSingleCrystal_->setAxisTitle("energy (GeV)", 1);

      histo = "EBCLTE SC seed occupancy map trigger tower binned";
      meSCSeedMapOccTT_ = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTT_->setAxisTitle("jphi", 1);
      meSCSeedMapOccTT_->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (high energy clusters) (crystal binned)";
      meSCSeedMapOccHighEne_ = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccHighEne_->setAxisTitle("jphi", 1);
      meSCSeedMapOccHighEne_->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC single crystal cluster occupancy map";
      meSCSeedMapOccSingleCrystal_ = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccSingleCrystal_->setAxisTitle("jphi", 1);
      meSCSeedMapOccSingleCrystal_->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing";
      meSCSeedTime_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTime_->setAxisTitle("seed crystal timing");

      histo = "EBCLTE SC seed crystal timing map trigger tower binned";
      meSCSeedMapTimeTT_ = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTT_->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTT_->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map module binned";
      meSCSeedMapTimeMod_ = dqmStore_->bookProfile2D(histo,histo,18,0,360,8,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeMod_->setAxisTitle("jphi",1);
      meSCSeedMapTimeMod_->setAxisTitle("jeta",2);

      histo = "EBCLTE SC seed crystal timing vs phi";
      meSCSeedTimeVsPhi_ = dqmStore_->bookProfile(histo, histo, 72, 0, 360, 78, -7, 7);
      meSCSeedTimeVsPhi_->setAxisTitle("jphi",1);
      meSCSeedTimeVsPhi_->setAxisTitle("seed crystal timing",2);

      histo = "EBCLTE SC seed crystal energy vs relative timing";
      meSCSeedTimeVsAmp_ = dqmStore_->bookProfile(histo, histo, 78, -7, 7, 200, 0, 1.8);
      meSCSeedTimeVsAmp_->setAxisTitle("seed crystal timing", 1);
      meSCSeedTimeVsAmp_->setAxisTitle("seed crystal energy (GeV)", 2);

      histo = "EBCLTE SC seed crystal timing EB -";
      meSCSeedTimeEBM_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEBM_->setAxisTitle("seed crystal timing");

      histo = "EBCLTE SC seed crystal timing EB +";
      meSCSeedTimeEBP_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEBP_->setAxisTitle("seed crystal timing");

      histo = "EBCLTE SC seed crystal timing EB - top";
      meSCSeedTimeEBMTop_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEBMTop_->setAxisTitle("seed crystal timing", 1);

      histo = "EBCLTE SC seed crystal timing EB + top";
      meSCSeedTimeEBPTop_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEBPTop_->setAxisTitle("seed crystal timing", 1);

      histo = "EBCLTE SC seed crystal timing EB - bottom";
      meSCSeedTimeEBMBot_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEBMBot_->setAxisTitle("seed crystal timing", 1);

      histo = "EBCLTE SC seed crystal timing EB + bottom";
      meSCSeedTimeEBPBot_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEBPBot_->setAxisTitle("seed crystal timing", 1);

      std::stringstream ss;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras/EBCLTE seed crystal timing per super module");
      for(int i=0;i!=36;++i) {
	ss.str("");
	if((i+1) <= 18){
	  ss << "EBCLTE SC seed crystal timing EB - " << i+1;
	  histo = ss.str();
	}
	else{
	  ss << "EBCLTE SC seed crystal timing EB + " << i+1-18;
	  histo = ss.str();
	}
	 meSCSeedTimePerFed_[i] = dqmStore_->book1D(histo,histo,78,0.,10.);
	 meSCSeedTimePerFed_[i]->setAxisTitle("seed crystal timing", 1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras");

      histo = "EBCLTE SC seed occupancy map (CSC exclusive triggered)";
      meSCSeedMapOccTrgExcl_[0] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrgExcl_[0]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExcl_[0]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (DT exclusive triggered)";
      meSCSeedMapOccTrgExcl_[1] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrgExcl_[1]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExcl_[1]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (ECAL exclusive triggered)";
      meSCSeedMapOccTrgExcl_[2] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrgExcl_[2]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExcl_[2]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (HCAL exclusive triggered)";
      meSCSeedMapOccTrgExcl_[3] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrgExcl_[3]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExcl_[3]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (RPC exclusive triggered)";
      meSCSeedMapOccTrgExcl_[4] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrgExcl_[4]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExcl_[4]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (CSC triggered)";
      meSCSeedMapOccTrg_[0] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrg_[0]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrg_[0]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (DT triggered)";
      meSCSeedMapOccTrg_[1] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrg_[1]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrg_[1]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (ECAL triggered)";
      meSCSeedMapOccTrg_[2] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrg_[2]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrg_[2]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (HCAL triggered)";
      meSCSeedMapOccTrg_[3] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrg_[3]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrg_[3]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (RPC triggered)";
      meSCSeedMapOccTrg_[4] = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOccTrg_[4]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrg_[4]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (CSC triggered) module binned";
      meSCSeedMapTimeTrgMod_[0] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgMod_[0]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgMod_[0]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (DT triggered) module binned";
      meSCSeedMapTimeTrgMod_[1] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgMod_[1]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgMod_[1]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (ECAL triggered) module binned";
      meSCSeedMapTimeTrgMod_[2] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgMod_[2]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgMod_[2]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (HCAL triggered) module binned";
      meSCSeedMapTimeTrgMod_[3] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgMod_[3]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgMod_[3]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (RPC triggered) module binned";
      meSCSeedMapTimeTrgMod_[4] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgMod_[4]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgMod_[4]->setAxisTitle("jeta", 2);

#endif

      histo = "EBCLTE SC size (crystal) vs energy (GeV)";
      meSCSizCrystalVsEne_ = dqmStore_->bookProfile(histo,histo,200,0.,10.,150,0,150);
      meSCSizCrystalVsEne_->setAxisTitle("energy (GeV)", 1);
      meSCSizCrystalVsEne_->setAxisTitle("super cluster size (crystal)", 2);

      histo = "EBCLTE SC seed occupancy map";
      meSCSeedMapOcc_ = dqmStore_->book2D(histo,histo,360,0,360,170,-85,85);
      meSCSeedMapOcc_->setAxisTitle("jphi", 1);
      meSCSeedMapOcc_->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (high energy clusters) trigger tower binned";
      meSCSeedMapOccHighEneTT_ = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccHighEneTT_->setAxisTitle("jphi", 1);
      meSCSeedMapOccHighEneTT_->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (CSC triggered) trigger tower binned";
      meSCSeedMapOccTrgTT_[0] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgTT_[0]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgTT_[0]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (DT triggered) trigger tower binned";
      meSCSeedMapOccTrgTT_[1] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgTT_[1]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgTT_[1]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (ECAL triggered) trigger tower binned";
      meSCSeedMapOccTrgTT_[2] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgTT_[2]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgTT_[2]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (HCAL triggered) trigger tower binned";
      meSCSeedMapOccTrgTT_[3] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgTT_[3]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgTT_[3]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (RPC triggered) trigger tower binned";
      meSCSeedMapOccTrgTT_[4] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgTT_[4]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgTT_[4]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (CSC exclusive triggered) trigger tower binned";
      meSCSeedMapOccTrgExclTT_[0] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgExclTT_[0]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExclTT_[0]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (DT exclusive triggered) trigger tower binned";
      meSCSeedMapOccTrgExclTT_[1] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgExclTT_[1]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExclTT_[1]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (ECAL exclusive triggered) trigger tower binned";
      meSCSeedMapOccTrgExclTT_[2] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgExclTT_[2]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExclTT_[2]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (HCAL exclusive triggered) trigger tower binned";
      meSCSeedMapOccTrgExclTT_[3] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgExclTT_[3]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExclTT_[3]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed occupancy map (RPC exclusive triggered) trigger tower binned";
      meSCSeedMapOccTrgExclTT_[4] = dqmStore_->book2D(histo,histo,72,0,360,34,-85,85);
      meSCSeedMapOccTrgExclTT_[4]->setAxisTitle("jphi", 1);
      meSCSeedMapOccTrgExclTT_[4]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (CSC exclusive triggered) trigger tower binned";
      meSCSeedMapTimeTrgTT_[0] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgTT_[0]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgTT_[0]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (DT exclusive triggered) trigger tower binned";
      meSCSeedMapTimeTrgTT_[1] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgTT_[1]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgTT_[1]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (ECAL exclusive triggered) trigger tower binned";
      meSCSeedMapTimeTrgTT_[2] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgTT_[2]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgTT_[2]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (HCAL exclusive triggered) trigger tower binned";
      meSCSeedMapTimeTrgTT_[3] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgTT_[3]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgTT_[3]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing map (RPC exclusive triggered) trigger tower binned";
      meSCSeedMapTimeTrgTT_[4] = dqmStore_->bookProfile2D(histo,histo,72,0,360,34,-85,85,78,0.,10.,"s");
      meSCSeedMapTimeTrgTT_[4]->setAxisTitle("jphi", 1);
      meSCSeedMapTimeTrgTT_[4]->setAxisTitle("jeta", 2);

      histo = "EBCLTE SC seed crystal timing (CSC triggered)";
      meSCSeedTimeTrg_[0] = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeTrg_[0]->setAxisTitle("seed crystal timing",1);

      histo = "EBCLTE SC seed crystal timing (DT triggered)";
      meSCSeedTimeTrg_[1] = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeTrg_[1]->setAxisTitle("seed crystal timing",1);

      histo = "EBCLTE SC seed crystal timing (ECAL triggered)";
      meSCSeedTimeTrg_[2] = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeTrg_[2]->setAxisTitle("seed crystal timing",1);

      histo = "EBCLTE SC seed crystal timing (HCAL triggered)";
      meSCSeedTimeTrg_[3] = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeTrg_[3]->setAxisTitle("seed crystal timing",1);

      histo = "EBCLTE SC seed crystal timing (RPC triggered)";
      meSCSeedTimeTrg_[4] = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeTrg_[4]->setAxisTitle("seed crystal timing",1);

      histo = "EBCLTE triggers";
      meTrg_ = dqmStore_->book1D(histo, histo, 5, 0, 5);
      meTrg_->setAxisTitle("triggers");

      histo = "EBCLTE exclusive triggers";
      meTrgExcl_ = dqmStore_->book1D(histo, histo, 5, 0, 5);
      meTrgExcl_->setAxisTitle("triggers");

   }

}

void EBClusterTaskExtras::cleanup(void){

   if ( ! init_ ) return;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras");

#ifndef EBCLUSTERTASKEXTRAS_DQMOFFLINE
      if ( meSCSizCrystal_ ) dqmStore_->removeElement( meSCSizCrystal_->getName() );
      meSCSizCrystal_ = 0;

      if ( meSCSizBC_ ) dqmStore_->removeElement( meSCSizBC_->getName() );
      meSCSizBC_ = 0;

      if ( meSCSizPhi_ ) dqmStore_->removeElement( meSCSizPhi_->getName() );
      meSCSizPhi_ = 0;

      if ( meSCSeedEne_ ) dqmStore_->removeElement( meSCSeedEne_->getName() );
      meSCSeedEne_ = 0;

      if ( meSCEne2_ ) dqmStore_->removeElement( meSCEne2_->getName() );
      meSCEne2_ = 0;

      if ( meSCEneLow_ ) dqmStore_->removeElement( meSCEneLow_->getName() );
      meSCEneLow_ = 0;

      if ( meSCEneHigh_ ) dqmStore_->removeElement( meSCEneHigh_->getName() );
      meSCEneHigh_ = 0;

      if ( meSCEneSingleCrystal_ ) dqmStore_->removeElement( meSCEneSingleCrystal_->getName() );
      meSCEneSingleCrystal_ = 0;

      if ( meSCSeedMapOccTT_ ) dqmStore_->removeElement( meSCSeedMapOccTT_->getName() );
      meSCSeedMapOccTT_ = 0;

      if ( meSCSeedMapOccHighEne_ ) dqmStore_->removeElement( meSCSeedMapOccHighEne_->getName() );
      meSCSeedMapOccHighEne_ = 0;

      if ( meSCSeedMapOccSingleCrystal_ ) dqmStore_->removeElement( meSCSeedMapOccSingleCrystal_->getName() );
      meSCSeedMapOccSingleCrystal_ = 0;

      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras/EBCLTE seed crystal timing per super module");
      for(int i=0; i!=36; ++i) {
	 if( meSCSeedTimePerFed_[i] ) dqmStore_->removeElement( meSCSeedTimePerFed_[i]->getName() );
	 meSCSeedTimePerFed_[i] = 0;
      }
      dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTaskExtras");

      if ( meSCSeedTime_ ) dqmStore_->removeElement( meSCSeedTime_->getName() );
      meSCSeedTime_ = 0;
      if ( meSCSeedMapTimeTT_ ) dqmStore_->removeElement( meSCSeedMapTimeTT_->getName() );
      meSCSeedMapTimeTT_ = 0;
      if ( meSCSeedMapTimeMod_ ) dqmStore_->removeElement( meSCSeedMapTimeMod_->getName() );
      meSCSeedMapTimeMod_ = 0;
      if ( meSCSeedTimeVsPhi_ ) dqmStore_->removeElement( meSCSeedTimeVsPhi_->getName() );
      meSCSeedTimeVsPhi_ = 0;
      if ( meSCSeedTimeVsAmp_ ) dqmStore_->removeElement( meSCSeedTimeVsAmp_->getName() );
      meSCSeedTimeVsAmp_ = 0;
      if ( meSCSeedTimeEBM_ ) dqmStore_->removeElement( meSCSeedTimeEBM_->getName() );
      meSCSeedTimeEBM_ = 0;
      if ( meSCSeedTimeEBP_ ) dqmStore_->removeElement( meSCSeedTimeEBP_->getName() );
      meSCSeedTimeEBP_ = 0;
      if ( meSCSeedTimeEBMTop_ ) dqmStore_->removeElement( meSCSeedTimeEBMTop_->getName() );
      meSCSeedTimeEBMTop_ = 0;
      if ( meSCSeedTimeEBPTop_ ) dqmStore_->removeElement( meSCSeedTimeEBPTop_->getName() );
      meSCSeedTimeEBPTop_ = 0;
      if ( meSCSeedTimeEBMBot_ ) dqmStore_->removeElement( meSCSeedTimeEBMBot_->getName() );
      meSCSeedTimeEBMBot_ = 0;
      if ( meSCSeedTimeEBPBot_ ) dqmStore_->removeElement( meSCSeedTimeEBPBot_->getName() );
      meSCSeedTimeEBPBot_ = 0;

      for(int i=0;i!=5+ii) {
	 if ( meSCSeedMapOccTrg_[i] ) dqmStore_->removeElement( meSCSeedMapOccTrg_[i]->getName() );
	 meSCSeedMapOccTrg_[i] = 0;
	 if ( meSCSeedMapOccTrgExcl_[i] ) dqmStore_->removeElement( meSCSeedMapOccTrgExcl_[i]->getName() );
	 meSCSeedMapOccTrgExcl_[i] = 0;
	 if ( meSCSeedMapTimeTrgMod_[i] ) dqmStore_->removeElement( meSCSeedMapTimeTrgMod_[i]->getName() );
	 meSCSeedMapTimeTrgMod_[i] = 0;
      }
#endif

      if ( meSCSizCrystalVsEne_ ) dqmStore_->removeElement( meSCSizCrystalVsEne_->getName() );
      meSCSizCrystalVsEne_ = 0;


      if ( meSCSeedMapOcc_ ) dqmStore_->removeElement( meSCSeedMapOcc_->getName() );
      meSCSeedMapOcc_ = 0;

      if ( meSCSeedMapOccHighEneTT_ ) dqmStore_->removeElement( meSCSeedMapOccHighEneTT_->getName() );
      meSCSeedMapOccHighEneTT_ = 0;

      for(int i=0; i!=5; ++i) {
	 if ( meSCSeedMapOccTrgTT_[i] ) dqmStore_->removeElement( meSCSeedMapOccTrgTT_[i]->getName() );
	 meSCSeedMapOccTrgTT_[i] = 0;
	 if ( meSCSeedMapOccTrgExclTT_[i] ) dqmStore_->removeElement( meSCSeedMapOccTrgExclTT_[i]->getName() );
	 meSCSeedMapOccTrgExclTT_[i] = 0;

	 if ( meSCSeedMapTimeTrgTT_[i] ) dqmStore_->removeElement( meSCSeedMapTimeTrgTT_[i]->getName() );
	 meSCSeedMapTimeTrgTT_[i] = 0;

	 if ( meSCSeedTimeTrg_[i] ) dqmStore_->removeElement( meSCSeedTimeTrg_[i]->getName() );
	 meSCSeedTimeTrg_[i] = 0;
      }

      if ( meTrg_ ) dqmStore_->removeElement( meTrg_->getName() );
      meTrg_ = 0;

      if ( meTrgExcl_ ) dqmStore_->removeElement( meTrgExcl_->getName() );
      meTrgExcl_ = 0;


   }

   init_ = false;

}

void EBClusterTaskExtras::endJob(void){

   LogInfo("EBClusterTaskExtras") << "analyzed " << ievt_ << " events";

   if ( enableCleanup_ ) this->cleanup();

}

void EBClusterTaskExtras::analyze(const Event& e, const EventSetup& c) {

   using namespace std;

   if ( ! init_ ) this->setup();

   ievt_++;

   // --- Barrel Super Clusters ---

   Handle<SuperClusterCollection> pSuperClusters;

   if ( e.getByToken(SuperClusterCollection_, pSuperClusters) ) {

      //int nscc = pSuperClusters->size();

      //TLorentzVector sc1_p(0,0,0,0);
      //TLorentzVector sc2_p(0,0,0,0);

      for ( SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); ++sCluster ) {

	 // seed and shapes
	 edm::Handle< EcalRecHitCollection > pEBRecHits;
	 e.getByToken( EcalRecHitCollection_, pEBRecHits );
	 if ( pEBRecHits.isValid() ) {
	    const EcalRecHitCollection *ebRecHits = pEBRecHits.product();

	    // Find the seed rec hit
	    // <= CMSSW_3_0_X
	    //std::vector<DetId> sIds = sCluster->getHitsByDetId();
	    // >= CMSSW_3_1_X
	    std::vector< std::pair<DetId,float> > sIds = sCluster->hitsAndFractions();

	    EcalRecHitCollection::const_iterator seedItr = ebRecHits->begin();
	    EcalRecHitCollection::const_iterator secondItr = ebRecHits->begin();

	    // <= CMSSW_3_0_X
	    //for(std::vector<DetId>::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
	       //if(idItr->det() != DetId::Ecal) { continue; }
	       //EcalRecHitCollection::const_iterator hitItr = ebRecHits->find((*idItr));
	       // <= CMSSW_3_1_X
	       for(std::vector< std::pair<DetId,float> >::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
	        DetId id = idItr->first;
	       if(id.det() != DetId::Ecal) { continue; }
	       EcalRecHitCollection::const_iterator hitItr = ebRecHits->find(id);
	       if(hitItr == ebRecHits->end()) { continue; }
	       if(hitItr->energy() > secondItr->energy()) { secondItr = hitItr; }
	       if(hitItr->energy() > seedItr->energy()) { std::swap(seedItr,secondItr); }
	    }

	    EBDetId seedId = (EBDetId) seedItr->id();

	    // Prepare to fill maps
	    int ebeta = seedId.ieta();
	    int ebphi = seedId.iphi();
	    float xebeta = ebeta - 0.5 * seedId.zside();
	    float xebphi = ebphi - 0.5;

	    // get the gain info;
	    edm::ESHandle<EcalADCToGeVConstant> pAgc;
	    c.get<EcalADCToGeVConstantRcd>().get(pAgc);

	    vector<bool> triggers = determineTriggers(e,c);

#ifndef EBCLUSTERTASKEXTRAS_DQMOFFLINE
            float e2nd = secondItr->energy();
	    float eMax = seedItr->energy();
	    int ism = Numbers::iSM(seedId);

	    // energy, size
	    if(meSCEneLow_) meSCEneLow_->Fill( sCluster->energy() );
	    if(meSCEneHigh_) meSCEneHigh_->Fill( sCluster->energy() );
	    if(meSCSizBC_) meSCSizBC_->Fill( float(sCluster->clustersSize()) );
	    if(meSCSizPhi_) meSCSizPhi_->Fill(xebphi,sCluster->clustersSize());

	    if(meSCSizCrystal_) meSCSizCrystal_->Fill(sIds.size());
	    if(meSCSeedEne_) meSCSeedEne_->Fill(eMax);
	    if(meSCEne2_) meSCEne2_->Fill(eMax+e2nd);
	    //if(meSCEneVsEMax_) meSCEneVsEMax_->Fill(eMax,sCluster->energy());

	    if(meSCSeedMapOccTT_) meSCSeedMapOccTT_->Fill(xebphi, xebeta);
	    if(sIds.size() == 1) {
	       if(meSCEneSingleCrystal_) meSCEneSingleCrystal_->Fill(sCluster->energy());
	       if(meSCSeedMapOccSingleCrystal_) meSCSeedMapOccSingleCrystal_->Fill(xebphi, xebeta);
	    }

	    if(sCluster->energy() > 2) if(meSCSeedMapOccHighEne_) meSCSeedMapOccHighEne_->Fill(xebphi, xebeta);
	    if(pAgc.isValid()) {
	       const EcalADCToGeVConstant* agc = pAgc.product();
	       if(seedItr->energy() / agc->getEBValue() > 12) {
		  if(meSCSeedTimeVsAmp_) meSCSeedTimeVsAmp_->Fill(seedItr->time() + 5,sCluster->energy());
		  if(meSCSeedTime_) meSCSeedTime_->Fill(seedItr->time() + 5);
		  if (ism >=  1 && ism <= 18 && meSCSeedTimeEBM_)  meSCSeedTimeEBM_->Fill(seedItr->time() + 5);
		  if (ism >= 19 && ism <= 36 && meSCSeedTimeEBP_)  meSCSeedTimeEBP_->Fill(seedItr->time() + 5);
		  if (ism >=  4 && ism <=  7 && meSCSeedTimeEBMTop_)  meSCSeedTimeEBMTop_->Fill(seedItr->time() + 5);
		  if (ism >= 22 && ism <= 25 && meSCSeedTimeEBPTop_)  meSCSeedTimeEBPTop_->Fill(seedItr->time() + 5);
		  if (ism >= 13 && ism <= 16 && meSCSeedTimeEBMBot_)  meSCSeedTimeEBMBot_->Fill(seedItr->time() + 5);
		  if (ism >= 31 && ism <= 34 && meSCSeedTimeEBPBot_)  meSCSeedTimeEBPBot_->Fill(seedItr->time() + 5);
		  if(meSCSeedTimePerFed_[ism-1]) meSCSeedTimePerFed_[ism-1]->Fill(seedItr->time() + 5);
		  if(meSCSeedMapTimeTT_) meSCSeedMapTimeTT_->Fill(xebphi,xebeta,seedItr->time() + 5);
		  if(meSCSeedMapTimeMod_) meSCSeedMapTimeMod_->Fill(xebphi,xebeta,seedItr->time() + 5);
		  if(meSCSeedTimeVsPhi_) meSCSeedTimeVsPhi_->Fill(xebphi,seedItr->time() + 5);
	       }
	    }
	    else {
	       LogWarning("EBClusterTaskExtras") << "EcalADCToGeVConstant not valid";
	    }

	    for(int i=0;i!=5;++i) {
	       if(triggers[i]) {
		  if(meSCSeedMapOccTrg_[i]) meSCSeedMapOccTrg_[i]->Fill(xebphi, xebeta);
		  bool isExclusive = true;
		  for(int j=0;j!=5;++j) {
		     if(pAgc.isValid()) 
			if(meSCSeedMapTimeTrgMod_[i]) meSCSeedMapTimeTrgMod_[i]->Fill(xebphi, xebeta, seedItr->time() + 5);
		     if(j != i && triggers[j])
			isExclusive = false;
		  }
		  if(isExclusive) if(meSCSeedMapOccTrgExcl_[i]) meSCSeedMapOccTrgExcl_[i]->Fill(xebphi, xebeta);
	       }
	    }
#endif
	    if(meSCSizCrystalVsEne_) meSCSizCrystalVsEne_->Fill(sCluster->energy(),sIds.size());

	    if(meSCSeedMapOcc_) meSCSeedMapOcc_->Fill(xebphi, xebeta);

	    if(sCluster->energy() > 2) if(meSCSeedMapOccHighEneTT_) meSCSeedMapOccHighEneTT_->Fill(xebphi, xebeta);
	    

	    for(int i=0;i!=5;++i) {
	       if(triggers[i]) {
		  if(meSCSeedMapOccTrgTT_[i]) meSCSeedMapOccTrgTT_[i]->Fill(xebphi, xebeta);
		  if(meTrg_) meTrg_->Fill(i);

		  if(pAgc.isValid()) {
		     if(meSCSeedMapTimeTrgTT_[i]) meSCSeedMapTimeTrgTT_[i]->Fill(xebphi, xebeta, seedItr->time() + 5);
		     if(meSCSeedTimeTrg_[i]) meSCSeedTimeTrg_[i]->Fill(seedItr->time() + 5);
		  }
		  else {
		     LogWarning("EBClusterTaskExtras") << "EcalADCToGeVConstant not valid";
		  }

		  bool isExclusive = true;
		  for(int j=0;j!=5;++j) {
		     if(j != i && triggers[j])
			isExclusive = false;
		  }

		  if(isExclusive) if(meTrgExcl_) meTrgExcl_->Fill(i);
	       }
	    }
	 }
	 else {
	    LogWarning("EBClusterTaskExtras") << "EcalRecHitCollection not available";
	 }

      }
   } else {

     //      LogWarning("EBClusterTaskExtras") << "SuperClusterCollection not available";

   }

}

std::vector<bool> 
EBClusterTaskExtras::determineTriggers(const edm::Event& iEvent, const edm::EventSetup& eventSetup) {

   using namespace edm;
   std::vector<bool> l1Triggers; //CSC,DT,HCAL,ECAL,RPC
   //0 , 1 , 2 , 3  , 4
   for(int i=0;i<5;i++)
      l1Triggers.push_back(false);

   // get the GMTReadoutCollection
   edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
   iEvent.getByToken(l1GMTReadoutRecToken_,gmtrc_handle);
   L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
   if (!(gmtrc_handle.isValid()))  
   {
      LogWarning("EcalCosmicsHists") << "l1MuGMTReadoutCollection" << " not available";
      return l1Triggers;
   }  
   // get hold of L1GlobalReadoutRecord
   edm::Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
   iEvent.getByToken(l1GTReadoutRecToken_,L1GTRR);

   //Ecal
   edm::ESHandle<L1GtTriggerMenu> menuRcd;
   eventSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
   const L1GtTriggerMenu* menu = menuRcd.product();
   // Get dWord after masking disabled bits
   const DecisionWord dWord = L1GTRR->decisionWord();

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

   l1Triggers[ECAL_TRIGGER] = l1SingleEG2 || l1SingleEG5 || l1SingleEG8 || l1SingleEG10 || l1SingleEG12 || l1SingleEG15
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
      if(igmtrr->getBxInEvent()==0 && idt>0) l1Triggers[DT_TRIGGER] = true;

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
      if(igmtrr->getBxInEvent()==0 && irpcb>0) l1Triggers[RPC_TRIGGER] = true;

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
      if(igmtrr->getBxInEvent()==0 && icsc>0) l1Triggers[CSC_TRIGGER] = true;
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
      if(ibx==0 && hcal_top && hcal_bot) l1Triggers[HCAL_TRIGGER]=true;
   }     
   return l1Triggers;
}

DEFINE_FWK_MODULE(EBClusterTaskExtras);
