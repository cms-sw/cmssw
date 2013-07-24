/*
 * \file EEClusterTaskExtras.cc
 *
 * $Date: 2012/06/28 12:15:16 $
 * $Revision: 1.10 $
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
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQMOffline/Ecal/interface/EEClusterTaskExtras.h>

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EEClusterTaskExtras::EEClusterTaskExtras(const ParameterSet& ps){

   init_ = false;

   dqmStore_ = Service<DQMStore>().operator->();

   prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

   enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

   mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

   // parameters...
   SuperClusterCollection_ = ps.getParameter<edm::InputTag>("SuperClusterCollection");
   EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
   l1GTReadoutRecTag_ = ps.getParameter<edm::InputTag>("l1GlobalReadoutRecord");
   l1GMTReadoutRecTag_ = ps.getParameter<edm::InputTag>("l1GlobalMuonReadoutRecord");

   // histograms...
#ifndef EECLUSTERTASKEXTRAS_DQMOFFLINE
   meSCSizCrystal_ = 0;
   meSCSizBC_ = 0;

   meSCSeedEne_ = 0;
   meSCEne2_ = 0;
   meSCEneLow_ = 0;
   meSCEneHigh_ = 0;
   meSCEneSingleCrystal_ = 0;

   for(int i=0;i!=2;++) {
      meSCSeedMapOccSingleCrystal_[i] = 0;
      meSCSeedMapOccSC_[i] = 0;
      meSCSeedMapOccHighEne_[i] = 0;
      meSCSeedMapTimeSC_[i] = 0;
      for(int j=0;j!=5;++j) {
	 meSCSeedMapOccTrg_[i][j] = 0;
	 meSCSeedMapOccTrgExcl_[i][j] = 0;
      }
   }

   meSCSeedTime_ = 0;
   meSCSeedTimeVsAmp_ = 0;
   meSCSeedTimeEEM_ = 0;
   meSCSeedTimeEEP_ = 0;
   for(int i=0;i!=18;++i)
      meSCSeedTimePerFed_[i] = 0;
#endif

  meSCSizCrystalVsEne_ = 0;

   for(int i=0;i!=2;++i) {
      meSCSeedMapOcc_[i] = 0;
      meSCSeedMapOccHighEneSC_[i] = 0;
      for(int j=0;j!=5;++j) {
	 meSCSeedMapTimeTrgSC_[i][j] = 0;
	 meSCSeedMapOccTrgSC_[i][j] = 0;
	 meSCSeedMapOccTrgExclSC_[i][j] = 0;
      }
   }

}

EEClusterTaskExtras::~EEClusterTaskExtras(){

}

void EEClusterTaskExtras::beginJob(){

   ievt_ = 0;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras");
      dqmStore_->rmdir(prefixME_ + "/EEClusterTaskExtras");
   }

}

void EEClusterTaskExtras::beginRun(const Run& r, const EventSetup& c) {

   Numbers::initGeometry(c, false);

   if ( ! mergeRuns_ ) this->reset();

}

void EEClusterTaskExtras::endRun(const Run& r, const EventSetup& c) {

}

void EEClusterTaskExtras::reset(void) {
#ifndef EECLUSTERTASKEXTRAS_DQMOFFLINE
   if ( meSCSizCrystal_ ) meSCSizCrystal_->Reset();
   if ( meSCSizBC_ ) meSCSizBC_->Reset(); 

   if ( meSCSeedEne_ ) meSCSeedEne_->Reset();
   if ( meSCEne2_ ) meSCEne2_->Reset();
   if ( meSCEneLow_ ) meSCEneLow_->Reset();
   if ( meSCEneHigh_ ) meSCEneHigh_->Reset();
   if ( meSCEneSingleCrystal_ ) meSCEneSingleCrystal_->Reset();

   for(int i=0;i!=2;++i) {
      if ( meSCSeedMapOccSingleCrystal_[i] ) meSCSeedMapOccSingleCrystal_[i]->Reset();
      if ( meSCSeedMapOccSC_[i] ) meSCSeedMapOccSC_[i]->Reset();
      if ( meSCSeedMapOccHighEne_[i] ) meSCSeedMapOccHighEne_[i]->Reset();
      if ( meSCSeedMapTimeSC_[i] ) meSCSeedMapTimeSC_[i]->Reset();
      for(int j=0;j!=5;++j) {
	 if ( meSCSeedMapOccTrg_[i][j] ) meSCSeedMapOccTrg_[i][j]->Reset();
	 if ( meSCSeedMapOccTrgExcl_[i][j] ) meSCSeedMapOccTrgExcl_[i][j]->Reset();
      }
   }

   if ( meSCSeedTime_ ) meSCSeedTime_->Reset();
   if ( meSCSeedTimeVsAmp_ ) meSCSeedTimeVsAmp_->Reset();
   if ( meSCSeedTimeEEM_ ) meSCSeedTimeEEM_->Reset();
   if ( meSCSeedTimeEEP_ ) meSCSeedTimeEEP_->Reset();

   for(int i=0;i!=18;++i) 
      if ( meSCSeedTimePerFed_[i] ) meSCSeedTimePerFed_[i]->Reset();
#endif

   if ( meSCSizCrystalVsEne_ ) meSCSizCrystalVsEne_->Reset();

   for(int i=0;i!=2;++i) {
      if ( meSCSeedMapOcc_[i] ) meSCSeedMapOcc_[i]->Reset();
      if ( meSCSeedMapOccHighEneSC_[i] ) meSCSeedMapOccHighEneSC_[i]->Reset();
      for(int j=0; j!=5; ++j) {
	 if ( meSCSeedMapOccTrgSC_[i][j] ) meSCSeedMapOccTrgSC_[i][j]->Reset();
	 if ( meSCSeedMapOccTrgExclSC_[i][j] ) meSCSeedMapOccTrgExclSC_[i][j]->Reset();
	 if ( meSCSeedMapTimeTrgSC_[i][j] ) meSCSeedMapTimeTrgSC_[i][j]->Reset();
      }
   }
   
}

void EEClusterTaskExtras::setup(void){

   init_ = true;

   std::string histo;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras");

#ifndef EECLUSTERTASKEXTRAS_DQMOFFLINE
      // Cluster hists
      histo = "EECLTE SC size (crystal)";
      meSCSizCrystal_ = dqmStore_->book1D(histo,histo,150,0,150);
      meSCSizCrystal_->setAxisTitle("super cluster size (crystal)", 1);

      histo = "EECLTE SC size (basic clusters)";
      meSCSizBC_ = dqmStore_->book1D(histo,histo,20,0,20);
      meSCSizBC_->setAxisTitle("super cluster size (basic clusters)", 1);

      histo = "EECLTE SC energy";
      meSCSeedEne_ = dqmStore_->book1D(histo,histo,200,0,1.8);
      meSCSeedEne_->setAxisTitle("energy (GeV)", 1);

      histo = "EECLTE SC + highest neighbor energy";
      meSCEne2_ = dqmStore_->book1D(histo,histo,200,0,1.8);
      meSCEne2_->setAxisTitle("+ highest neighbor energy (GeV)", 1);

      histo = "EECLTE SC energy low scale";
      meSCEneLow_ = dqmStore_->book1D(histo,histo,200,0,1.8);
      meSCEneLow_->setAxisTitle("energy (GeV)", 1);

      histo = "EECLTE SC energy high scale";
      meSCEneHigh_ = dqmStore_->book1D(histo,histo,200,0,200);
      meSCEneHigh_->setAxisTitle("energy (GeV)", 1);

      histo = "EECLTE SC single crystal cluster energy (GeV)";
      meSCEneSingleCrystal_ = dqmStore_->book1D(histo,histo,200,0,200);
      meSCEneSingleCrystal_->setAxisTitle("energy (GeV)", 1);

      histo = "EECLTE SC seed occupancy map super crystal binned EE -";
      meSCSeedMapOccSC_[0] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccSC_[0]->setAxisTitle("jx", 1);
      meSCSeedMapOccSC_[0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (high energy clusters)";
      meSCSeedMapOccHighEne_[0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccHighEne_[0]->setAxisTitle("jx", 1);
      meSCSeedMapOccHighEne_[0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC single crystal cluster seed occupancy map EE -";
      meSCSeedMapOccSingleCrystal_[0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccSingleCrystal_[0]->setAxisTitle("jx", 1);
      meSCSeedMapOccSingleCrystal_[0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (CSC triggered)";
      meSCSeedMapOccTrg_[0][0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[0][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[0][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (DT triggered)";
      meSCSeedMapOccTrg_[0][1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[0][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[0][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (ECAL triggered)";
      meSCSeedMapOccTrg_[0][2] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[0][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[0][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (HCAL triggered)";
      meSCSeedMapOccTrg_[0][3] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[0][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[0][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (RPC triggered)";
      meSCSeedMapOccTrg_[0][4] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[0][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[0][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (CSC exclusive triggered)";
      meSCSeedMapOccTrgExcl_[0][0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[0][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[0][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (DT exclusive triggered)";
      meSCSeedMapOccTrgExcl_[0][1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[0][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[0][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (ECAL exclusive triggered)";
      meSCSeedMapOccTrgExcl_[0][2] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[0][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[0][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (HCAL exclusive triggered)";
      meSCSeedMapOccTrgExcl_[0][3] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[0][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[0][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (RPC exclusive triggered)";
      meSCSeedMapOccTrgExcl_[0][4] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[0][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[0][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map super crystal binned EE +";
      meSCSeedMapOccSC_[1] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccSC_[1]->setAxisTitle("jx", 1);
      meSCSeedMapOccSC_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (high energy clusters)";
      meSCSeedMapOccHighEne_[1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccHighEne_[1]->setAxisTitle("jx", 1);
      meSCSeedMapOccHighEne_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC single crystal cluster seed occupancy map EE +";
      meSCSeedMapOccSingleCrystal_[1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccSingleCrystal_[1]->setAxisTitle("jx", 1);
      meSCSeedMapOccSingleCrystal_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (CSC triggered)";
      meSCSeedMapOccTrg_[1][0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[1][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[1][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (DT triggered)";
      meSCSeedMapOccTrg_[1][1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[1][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[1][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (ECAL triggered)";
      meSCSeedMapOccTrg_[1][2] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[1][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[1][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (HCAL triggered)";
      meSCSeedMapOccTrg_[1][3] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[1][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[1][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (RPC triggered)";
      meSCSeedMapOccTrg_[1][4] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrg_[1][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrg_[1][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (CSC exclusive triggered)";
      meSCSeedMapOccTrgExcl_[1][0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[1][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[1][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (DT exclusive triggered)";
      meSCSeedMapOccTrgExcl_[1][1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[1][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[1][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (ECAL exclusive triggered)";
      meSCSeedMapOccTrgExcl_[1][2] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[1][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[1][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (HCAL exclusive triggered)";
      meSCSeedMapOccTrgExcl_[1][3] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[1][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[1][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (RPC exclusive triggered)";
      meSCSeedMapOccTrgExcl_[1][4] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOccTrgExcl_[1][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExcl_[1][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + super crystal binned";
      meSCSeedMapTimeSC_[1] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeSC_[1]->setAxisTitle("jx", 1);
      meSCSeedMapTimeSC_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + super crystal binned";
      meSCSeedMapTimeSC_[1] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeSC_[1]->setAxisTitle("jx", 1);
      meSCSeedMapTimeSC_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC relative timing";
      meSCSeedTime_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTime_->setAxisTitle("seed crystal timing");

      histo = "EECLTE SC relative timing vs amplitude";
      meSCSeedTimeVsAmp_ = dqmStore_->bookProfile(histo, histo, 78, -7, 7, 200, 0, 1.8);
      meSCSeedTimeVsAmp_->setAxisTitle("seed crystal timing", 1);
      meSCSeedTimeVsAmp_->setAxisTitle("energy (GeV)", 2);

      histo = "EECLTE SC relative timing EE -";
      meSCSeedTimeEEM_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEEM_->setAxisTitle("seed crystal timing");

      histo = "EECLTE SC relative timing EE +";
      meSCSeedTimeEEP_ = dqmStore_->book1D(histo,histo,78,0.,10.);
      meSCSeedTimeEEP_->setAxisTitle("seed crystal timing");

      std::stringstream ss;
      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras/EECLTE timing per super module");
      for(int i=0;i!=18;++i) {
	ss.str("");
	if((i+1) <= 9){
	  ss << "EECLTE SC timing EE - " << i+1;
	  histo = ss.str();
	}
	else{
	  ss << "EECLTE SC timing EE + " << i+1-9;
	  histo = ss.str();
	}
	 meSCSeedTimePerFed_[i] = dqmStore_->book1D(histo,histo,78,0.,10.);
	 meSCSeedTimePerFed_[i]->setAxisTitle("seed crystal timing", 1);
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras");

#endif

      histo = "EECLTE SC size (crystal) vs energy (GeV)";
     meSCSizCrystalVsEne_ = dqmStore_->bookProfile(histo,histo,200,0.,10.,150,0,150);
     meSCSizCrystalVsEne_->setAxisTitle("energy (GeV)", 1);
     meSCSizCrystalVsEne_->setAxisTitle("super cluster size (crystal)", 2);

      histo = "EECLTE SC seed occupancy map EE -";
      meSCSeedMapOcc_[0] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOcc_[0]->setAxisTitle("jx", 1);
      meSCSeedMapOcc_[0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (high energy clusters) super crystal binned";
      meSCSeedMapOccHighEneSC_[0] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccHighEneSC_[0]->setAxisTitle("jx", 1);
      meSCSeedMapOccHighEneSC_[0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (CSC triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[0][0] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[0][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[0][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (DT triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[0][1] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[0][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[0][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (ECAL triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[0][2] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[0][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[0][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (HCAL triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[0][3] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[0][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[0][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (RPC triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[0][4] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[0][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[0][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (CSC exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[0][0] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[0][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[0][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (DT exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[0][1] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[0][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[0][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (ECAL exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[0][2] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[0][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[0][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (HCAL exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[0][3] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[0][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[0][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE - (RPC exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[0][4] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[0][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[0][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE +";
      meSCSeedMapOcc_[1] = dqmStore_->book2D(histo,histo,100,0,100,100,0,100);
      meSCSeedMapOcc_[1]->setAxisTitle("jx", 1);
      meSCSeedMapOcc_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (high energy clusters) super crystal binned";
      meSCSeedMapOccHighEneSC_[1] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccHighEneSC_[1]->setAxisTitle("jx", 1);
      meSCSeedMapOccHighEneSC_[1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (CSC triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[1][0] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[1][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[1][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (DT triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[1][1] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[1][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[1][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (ECAL triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[1][2] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[1][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[1][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (HCAL triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[1][3] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[1][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[1][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (RPC triggered) super crystal binned";
      meSCSeedMapOccTrgSC_[1][4] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgSC_[1][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgSC_[1][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (CSC exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[1][0] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[1][0]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[1][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (DT exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[1][1] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[1][1]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[1][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (ECAL exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[1][2] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[1][2]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[1][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (HCAL exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[1][3] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[1][3]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[1][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed occupancy map EE + (RPC exclusive triggered) super crystal binned";
      meSCSeedMapOccTrgExclSC_[1][4] = dqmStore_->book2D(histo,histo,20,0,100,20,0,100);
      meSCSeedMapOccTrgExclSC_[1][4]->setAxisTitle("jx", 1);
      meSCSeedMapOccTrgExclSC_[1][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE - (CSC exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[0][0] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[0][0]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[0][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE - (DT exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[0][1] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[0][1]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[0][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE - (ECAL exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[0][2] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[0][2]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[0][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE - (HCAL exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[0][3] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[0][3]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[0][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE - (RPC exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[0][4] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[0][4]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[0][4]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + (CSC exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[1][0] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[1][0]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[1][0]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + (DT exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[1][1] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[1][1]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[1][1]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + (ECAL exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[1][2] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[1][2]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[1][2]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + (HCAL exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[1][3] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[1][3]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[1][3]->setAxisTitle("jy", 2);

      histo = "EECLTE SC seed crystal timing map EE + (RPC exclusive triggered) super crystal binned";
      meSCSeedMapTimeTrgSC_[1][4] = dqmStore_->bookProfile2D(histo,histo,20,0,100,20,0,100,78,0.,10.,"s");
      meSCSeedMapTimeTrgSC_[1][4]->setAxisTitle("jx", 1);
      meSCSeedMapTimeTrgSC_[1][4]->setAxisTitle("jy", 2);
   }
}

void EEClusterTaskExtras::cleanup(void){

   if ( ! init_ ) return;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras");

#ifndef EECLUSTERTASKEXTRAS_DQMOFFLINE
      if ( meSCSizCrystal_ ) dqmStore_->removeElement( meSCSizCrystal_->getName() );
      meSCSizCrystal_ = 0;
      if ( meSCSizBC_ ) dqmStore_->removeElement( meSCSizBC_->getName() );
      meSCSizBC_ = 0;

      if ( meSCSeedEne_ ) dqmStore_->removeElement( meSCSeedEne_->getName() );
      meSCSeedEne_ = 0;
      if ( meSCEne2_ ) dqmStore_->removeElement( meSCEne2_->getName() );
      meSCEne2_= 0;
      if ( meSCEneLow_ ) dqmStore_->removeElement( meSCEneLow_->getName() );
      meSCEneLow_ = 0;
      if ( meSCEneHigh_ ) dqmStore_->removeElement( meSCEneHigh_->getName() );
      meSCEneHigh_ = 0;
      if ( meSCEneSingleCrystal_ ) dqmStore_->removeElement( meSCEneSingleCrystal_->getName() );
      meSCEneSingleCrystal_ = 0;

      for(int i=0;i!=2;++i) {
      	 if ( meSCSeedMapOccSC_[i] ) dqmStore_->removeElement( meSCSeedMapOccSC_[i]->getName() );
	 meSCSeedMapOccSC_[i] = 0;
	 if ( meSCSeedMapOccHighEne_[i] ) dqmStore_->removeElement( meSCSeedMapOccHighEne_[i]->getName() );
	 meSCSeedMapOccHighEne_[i] = 0;
	 if ( meSCSeedMapOccSingleCrystal_[i] ) dqmStore_->removeElement( meSCSeedMapOccSingleCrystal_[i]->getName() );
	 meSCSeedMapOccSingleCrystal_[i] = 0;
	 if ( meSCSeedMapTimeSC_[i] ) dqmStore_->removeElement( meSCSeedMapTimeSC_[i]->getName() );
	 meSCSeedMapTimeSC_[i] = 0;
	 for(int j=0;j!=5;++j) {
	    if ( meSCSeedMapOccTrg_[i][j] ) dqmStore_->removeElement( meSCSeedMapOccTrg_[i][j]->getName() );
	    meSCSeedMapOccTrg_[i][j] = 0;
	    if ( meSCSeedMapOccTrgExcl_[i][j] ) dqmStore_->removeElement( meSCSeedMapOccTrgExcl_[i][j]->getName() );
	    meSCSeedMapOccTrgExcl_[i][j] = 0;
	 }
      }

      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras/EECLTE timing per super module");
      for(int i=0; i!=18; ++i) {
	 if( meSCSeedTimePerFed_[i] ) dqmStore_->removeElement( meSCSeedTimePerFed_[i]->getName() );
	 meSCSeedTimePerFed_[i] = 0;
      }
      dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTaskExtras");

      if ( meSCSeedTime_ ) dqmStore_->removeElement( meSCSeedTime_->getName() );
      meSCSeedTime_ = 0;
      if ( meSCSeedTimeVsAmp_ ) dqmStore_->removeElement( meSCSeedTimeVsAmp_->getName() );
      meSCSeedTimeVsAmp_ = 0;
      if ( meSCSeedTimeEEM_ ) dqmStore_->removeElement( meSCSeedTimeEEM_->getName() );
      meSCSeedTimeEEM_ = 0;
      if ( meSCSeedTimeEEP_ ) dqmStore_->removeElement( meSCSeedTimeEEP_->getName() );
      meSCSeedTimeEEP_ = 0;

#endif

      if (meSCSizCrystalVsEne_ ) dqmStore_->removeElement(meSCSizCrystalVsEne_->getName() );
     meSCSizCrystalVsEne_ = 0;

      for(int i=0;i!=2;++i) {
	 if ( meSCSeedMapOcc_[i] ) dqmStore_->removeElement( meSCSeedMapOcc_[i]->getName() );
	 meSCSeedMapOcc_[i] = 0;
	 if ( meSCSeedMapOccHighEneSC_[i] ) dqmStore_->removeElement( meSCSeedMapOccHighEneSC_[i]->getName() );
	 meSCSeedMapOccHighEneSC_[i] = 0;

	 for(int j=0; j!=5; ++j) {
	    if ( meSCSeedMapOccTrgSC_[i][j] ) dqmStore_->removeElement( meSCSeedMapOccTrgSC_[i][j]->getName() );
	    meSCSeedMapOccTrgSC_[i][j] = 0;
	    if ( meSCSeedMapOccTrgExclSC_[i][j] ) dqmStore_->removeElement( meSCSeedMapOccTrgExclSC_[i][j]->getName() );
	    meSCSeedMapOccTrgExclSC_[i][j] = 0;

	    if ( meSCSeedMapTimeTrgSC_[i][j] ) dqmStore_->removeElement( meSCSeedMapTimeTrgSC_[i][j]->getName() );
	    meSCSeedMapTimeTrgSC_[i][j] = 0;
	 }
      }

   }

   init_ = false;

}

void EEClusterTaskExtras::endJob(void){

   LogInfo("EEClusterTaskExtras") << "analyzed " << ievt_ << " events";

   if ( enableCleanup_ ) this->cleanup();

}

void EEClusterTaskExtras::analyze(const Event& e, const EventSetup& c) {

   using namespace std;

   if ( ! init_ ) this->setup();

   ievt_++;

   // --- Barrel Super Clusters ---

   Handle<SuperClusterCollection> pSuperClusters;

   if ( e.getByLabel(SuperClusterCollection_, pSuperClusters) ) {

      //int nscc = pSuperClusters->size();

      //TLorentzVector sc1_p(0,0,0,0);
      //TLorentzVector sc2_p(0,0,0,0);

      for ( SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); ++sCluster ) {

	 // seed and shapes
	 edm::Handle< EcalRecHitCollection > pEERecHits;
	 e.getByLabel( EcalRecHitCollection_, pEERecHits );
	 if ( pEERecHits.isValid() ) {
	    const EcalRecHitCollection *eeRecHits = pEERecHits.product();

	    // Find the seed rec hit
	    // <= CMSSW_3_0_X
	    //std::vector<DetId> sIds = sCluster->getHitsByDetId();
	    // >= CMSSW_3_1_X
	    std::vector< std::pair<DetId,float> > sIds = sCluster->hitsAndFractions();

	    EcalRecHitCollection::const_iterator seedItr = eeRecHits->begin();
	    EcalRecHitCollection::const_iterator secondItr = eeRecHits->begin();

	    // <= CMSSW_3_0_X
	    //for(std::vector<DetId>::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
	       //if(idItr->det() != DetId::Ecal) { continue; }
	       //EcalRecHitCollection::const_iterator hitItr = eeRecHits->find((*idItr));
	    // >= CMSSW_3_1_X
	    for(std::vector< std::pair<DetId,float> >::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
	       DetId id = idItr->first;
	       if(id.det() != DetId::Ecal) { continue; }
	       EcalRecHitCollection::const_iterator hitItr = eeRecHits->find(id);
	       if(hitItr == eeRecHits->end()) { continue; }
	       if(hitItr->energy() > secondItr->energy()) { secondItr = hitItr; }
	       if(hitItr->energy() > seedItr->energy()) { std::swap(seedItr,secondItr); }
	    }

	    EEDetId seedId = (EEDetId) seedItr->id();

	    // Prepare to fill maps
	    int ism = Numbers::iSM(seedId);
	    int eey = seedId.iy();
	    int eex = seedId.ix();
	    float xeey = eey - 0.5;
	    float xeex = eex - 0.5;

	    int side = (ism >=1 && ism <= 9) ? 0 : 1;

	    edm::ESHandle<EcalADCToGeVConstant> pAgc;
	    c.get<EcalADCToGeVConstantRcd>().get(pAgc);

	    vector<bool> triggers = determineTriggers(e,c);

#ifndef EECLUSTERTASKEXTRAS_DQMOFFLINE
	    float eMax, e2nd;
	    eMax = seedItr->energy();
	    e2nd = secondItr->energy();
	    // energy, size
	    if(meSCEneLow_) meSCEneLow_->Fill( sCluster->energy() );
	    if(meSCEneHigh_) meSCEneHigh_->Fill( sCluster->energy() );
	    if(meSCSizBC_) meSCSizBC_->Fill( float(sCluster->clustersSize()) );

	    if(meSCSizCrystal_) meSCSizCrystal_->Fill(sIds.size());
	    if(meSCSeedEne_) meSCSeedEne_->Fill(eMax);
	    if(meSCEne2_) meSCEne2_->Fill(eMax+e2nd);
	    //if(meSCEneVsEMax_) meSCEneVsEMax_->Fill(eMax,sCluster->energy());

	    if(meSCSeedMapOccSC_[side]) meSCSeedMapOccSC_[side]->Fill(xeex, xeey);

	    if(sCluster->energy() > 2) {
	       if(meSCSeedMapOccHighEne_[side]) meSCSeedMapOccHighEne_[side]->Fill(xeex, xeey);
	       if(meSCSeedMapOccHighEneSC_[side]) meSCSeedMapOccHighEneSC_[side]->Fill(xeex, xeey);
	    }
	    if(sIds.size() == 1) {
	       if(meSCEneSingleCrystal_) meSCEneSingleCrystal_->Fill(sCluster->energy());
	       if(meSCSeedMapOccSingleCrystal_[side]) meSCSeedMapOccSingleCrystal_[side]->Fill(xeex, xeey);
	    }

	    if(meSCSeedMapOcc_[side]) meSCSeedMapOcc_[side]->Fill(xeex, xeey);

	    if(pAgc.isValid()) {
	       const EcalADCToGeVConstant* agc = pAgc.product();
	       if(seedItr->energy() / agc->getEBValue() > 12) {
		  if(meSCSeedTime_) meSCSeedTime_->Fill(seedItr->time());
		  if(meSCSeedTimeVsAmp_) meSCSeedTimeVsAmp_->Fill(seedItr->time(),sCluster->energy());
		  if(!side)
		     if(meSCSeedTimeEEM_) meSCSeedTimeEEM_->Fill(seedItr->time());
		  if(side)
		     if(meSCSeedTimeEEP_) meSCSeedTimeEEP_->Fill(seedItr->time());
		  if(meSCSeedTimePerFed_[ism-1]) meSCSeedTimePerFed_[ism-1]->Fill(seedItr->time());
		  if(meSCSeedMapTimeSC_[side]) meSCSeedMapTimeSC_[side]->Fill(xeex,xeey,seedItr->time());

	       }
	    }
	    else {
	       LogWarning("EBClusterTaskExtras") << "EcalADCToGeVConstant not valid";
	    }
	    for(int i=0;i!=5;++i) {
	       if(triggers[i]) {
		  if(meSCSeedMapOccTrg_[side][i]) meSCSeedMapOccTrg_[side][i]->Fill(xeex, xeey);
		  bool isExclusive = true;
		  for(int j=0;j!=5;++j) {
		     if(j != i && triggers[j])
			isExclusive = false;
		  }
		  if(isExclusive) 
		     if(meSCSeedMapOccTrgExcl_[side][i]) meSCSeedMapOccTrgExcl_[side][i]->Fill(xeex, xeey);
	       }
	    }
#endif

	    if(meSCSizCrystalVsEne_) meSCSizCrystalVsEne_->Fill(sCluster->energy(),sIds.size());

	    for(int i=0;i!=5;++i) {
	       if(triggers[i]) {
		  if(meSCSeedMapOccTrgSC_[side][i]) meSCSeedMapOccTrgSC_[side][i]->Fill(xeex, xeey);

		  if(pAgc.isValid()) {
		     const EcalADCToGeVConstant* agc = pAgc.product();
		     if(seedItr->energy() / agc->getEBValue() > 12) {
			if(meSCSeedMapTimeTrgSC_[side][i]) meSCSeedMapTimeTrgSC_[side][i]->Fill(xeex, xeey, seedItr->time());
		     }
		  }
		  else {
		     LogWarning("EBClusterTaskExtras") << "EcalADCToGeVConstant not valid";
		  }

		  bool isExclusive = true;
		  for(int j=0;j!=5;++j) {
		     if(j != i && triggers[j])
			isExclusive = false;
		  }
		  if(isExclusive) 
		     if(meSCSeedMapOccTrgExclSC_[side][i]) meSCSeedMapOccTrgExclSC_[side][i]->Fill(xeex, xeey);
	       }
	    }
	 }
	 else {
	    LogWarning("EEClusterTaskExtras") << pEERecHits << " not available";
	 }
      }

   } else {

      LogWarning("EEClusterTaskExtras") << SuperClusterCollection_ << " not available";

   }

}

std::vector<bool> 
EEClusterTaskExtras::determineTriggers(const edm::Event& iEvent, const edm::EventSetup& eventSetup) {

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
      std::vector<int> valid_x; 
      if((psb.aData(4)&0x3f) >= 1) {valid_x.push_back( (psb.aData(4)>>10)&0x1f ); }
      if((psb.bData(4)&0x3f) >= 1) {valid_x.push_back( (psb.bData(4)>>10)&0x1f ); }
      if((psb.aData(5)&0x3f) >= 1) {valid_x.push_back( (psb.aData(5)>>10)&0x1f ); }
      if((psb.bData(5)&0x3f) >= 1) {valid_x.push_back( (psb.bData(5)>>10)&0x1f ); }
      std::vector<int>::const_iterator ix;
      for(ix=valid_x.begin(); ix!=valid_x.end(); ix++) {
	 //std::cout << "Found HCAL mip with x=" << *ix << " in bx wrt. L1A = " << ibx << std::endl;
	 if(*ix<9) hcal_top=true;
	 if(*ix>8) hcal_bot=true;
      }
      if(ibx==0 && hcal_top && hcal_bot) l1Triggers[HCAL_TRIGGER]=true;
   }     
   return l1Triggers;
}
