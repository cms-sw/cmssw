/*
 * \file EETrendTask.cc
 *
 * $Date: 2009/11/19 18:14:45 $
 * $Revision: 1.3 $
 * \author Dongwook Jang, Soon Yung Jun
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

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"


#include "DQM/EcalEndcapMonitorTasks/interface/EETrendTask.h"
#include "DQM/EcalCommon/interface/UtilFunctions.h"

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EETrendTask::EETrendTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // parameters...
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");

  // histograms...
  nEEDigiMinutely_ = 0;
  nEcalRecHitMinutely_ = 0;
  nFEDEEminusRawDataMinutely_ = 0;
  nFEDEEplusRawDataMinutely_ = 0;

  nEEDigiHourly_ = 0;
  nEcalRecHitHourly_ = 0;
  nFEDEEminusRawDataHourly_ = 0;
  nFEDEEplusRawDataHourly_ = 0;
}


EETrendTask::~EETrendTask(){
}


void EETrendTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETrendTask");
    dqmStore_->rmdir(prefixME_ + "/EETrendTask");
  }

}


void EETrendTask::beginRun(const Run& r, const EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  start_time_ = time(NULL);

}


void EETrendTask::endRun(const Run& r, const EventSetup& c) {

}


void EETrendTask::reset(void) {

  if(nEEDigiMinutely_) nEEDigiMinutely_->Reset();
  if(nEcalRecHitMinutely_) nEcalRecHitMinutely_->Reset();
  if(nFEDEEminusRawDataMinutely_) nFEDEEminusRawDataMinutely_->Reset();
  if(nFEDEEplusRawDataMinutely_) nFEDEEplusRawDataMinutely_->Reset();

  if(nEEDigiHourly_) nEEDigiHourly_->Reset();
  if(nEcalRecHitHourly_) nEcalRecHitHourly_->Reset();
  if(nFEDEEminusRawDataHourly_) nFEDEEminusRawDataHourly_->Reset();
  if(nFEDEEplusRawDataHourly_) nFEDEEplusRawDataHourly_->Reset();

}


void EETrendTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETrendTask");

    // minutely

    sprintf(histo, "AverageNumberOfEEDigiVs5Minutes");
    nEEDigiMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEEDigiMinutely_->setAxisTitle("Minutes", 1);
    nEEDigiMinutely_->setAxisTitle("Average Number of EEDigi / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfEcalRecHitVs5Minutes");
    nEcalRecHitMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEcalRecHitMinutely_->setAxisTitle("Minutes", 1);
    nEcalRecHitMinutely_->setAxisTitle("Average Number of EcalRecHit / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfFEDEEminusRawDataVs5Minutes");
    nFEDEEminusRawDataMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nFEDEEminusRawDataMinutely_->setAxisTitle("Minutes", 1);
    nFEDEEminusRawDataMinutely_->setAxisTitle("Average Number of FEDRawData in EE- / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfFEDEEplusRawDataVs5Minutes");
    nFEDEEplusRawDataMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nFEDEEplusRawDataMinutely_->setAxisTitle("Minutes", 1);
    nFEDEEplusRawDataMinutely_->setAxisTitle("Average Number of FEDRawData in EE+ / 5 minutes", 2);


    // hourly

    sprintf(histo, "AverageNumberOfEEDigiVs1Hour");
    nEEDigiHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEEDigiHourly_->setAxisTitle("Hours", 1);
    nEEDigiHourly_->setAxisTitle("Average Number of EEDigi / hour", 2);

    sprintf(histo, "AverageNumberOfEcalRecHitVs1Hour");
    nEcalRecHitHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEcalRecHitHourly_->setAxisTitle("Hours", 1);
    nEcalRecHitHourly_->setAxisTitle("Average Number of EcalRecHit / hour", 2);

    sprintf(histo, "AverageNumberOfFEDEEminusRawDataVs1Hour");
    nFEDEEminusRawDataHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nFEDEEminusRawDataHourly_->setAxisTitle("Hours", 1);
    nFEDEEminusRawDataHourly_->setAxisTitle("Average Number of FEDRawData in EE- / hour", 2);

    sprintf(histo, "AverageNumberOfFEDEEplusRawDataVs1Hour");
    nFEDEEplusRawDataHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nFEDEEplusRawDataHourly_->setAxisTitle("Hours", 1);
    nFEDEEplusRawDataHourly_->setAxisTitle("Average Number of FEDRawData in EE+ / hour", 2);

  }

}


void EETrendTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETrendTask");

    if(nEEDigiMinutely_) dqmStore_->removeElement( nEEDigiMinutely_->getName());
    nEEDigiMinutely_ = 0;
    if(nEcalRecHitMinutely_) dqmStore_->removeElement( nEcalRecHitMinutely_->getName());
    nEcalRecHitMinutely_ = 0;
    if(nFEDEEminusRawDataMinutely_) dqmStore_->removeElement( nFEDEEminusRawDataMinutely_->getName());
    nFEDEEminusRawDataMinutely_ = 0;
    if(nFEDEEplusRawDataMinutely_) dqmStore_->removeElement( nFEDEEplusRawDataMinutely_->getName());
    nFEDEEplusRawDataMinutely_ = 0;

    if(nEEDigiHourly_) dqmStore_->removeElement( nEEDigiHourly_->getName());
    nEEDigiHourly_ = 0;
    if(nEcalRecHitHourly_) dqmStore_->removeElement( nEcalRecHitHourly_->getName());
    nEcalRecHitHourly_ = 0;
    if(nFEDEEminusRawDataHourly_) dqmStore_->removeElement( nFEDEEminusRawDataHourly_->getName());
    nFEDEEminusRawDataHourly_ = 0;
    if(nFEDEEplusRawDataHourly_) dqmStore_->removeElement( nFEDEEplusRawDataHourly_->getName());
    nFEDEEplusRawDataHourly_ = 0;

  }

  init_ = false;

}


void EETrendTask::endJob(void){

  LogInfo("EETrendTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}


void EETrendTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --------------------------------------------------
  // Collect time information
  // --------------------------------------------------

  updateTime();

  long int minuteBinDiff = -1;
  long int minuteDiff = -1;
  ecaldqm::calcBins(5,60,start_time_,last_time_,current_time_,minuteBinDiff,minuteDiff);

  long int hourBinDiff = -1;
  long int hourDiff = -1;
  ecaldqm::calcBins(1,3600,start_time_,last_time_,current_time_,hourBinDiff,hourDiff);


  // --------------------------------------------------
  // EEDigiCollection
  // --------------------------------------------------
  int ndc = 0;
  Handle<EEDigiCollection> digis;
  if ( e.getByLabel(EEDigiCollection_, digis) ) ndc = digis->size();
  else LogWarning("EETrendTask") << EEDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEEDigiMinutely_->getTProfile(), minuteBinDiff);
  nEEDigiMinutely_->Fill(minuteDiff,ndc);
  
  ecaldqm::shift2Right(nEEDigiHourly_->getTProfile(), hourBinDiff);
  nEEDigiHourly_->Fill(hourDiff,ndc);


  // --------------------------------------------------
  // EcalRecHitCollection
  // --------------------------------------------------
  int nrhc = 0;
  Handle<EcalRecHitCollection> hits;
  if ( e.getByLabel(EcalRecHitCollection_, hits) ) nrhc = hits->size();
  else LogWarning("EETrendTask") << EcalRecHitCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalRecHitMinutely_->getTProfile(), minuteBinDiff);
  nEcalRecHitMinutely_->Fill(minuteDiff,nrhc);
  
  ecaldqm::shift2Right(nEcalRecHitHourly_->getTProfile(), hourBinDiff);
  nEcalRecHitHourly_->Fill(hourDiff,nrhc);


  // --------------------------------------------------
  // FEDRawDataCollection
  // --------------------------------------------------
  int nfedEEminus = 0;
  int nfedEEplus  = 0;

  // Barrel FEDs : 610 - 645
  // Endcap FEDs : 601-609 (EE-) and 646-654 (EE+) 
  int eem1 = 601;
  int eem2 = 609;
  int eep1 = 646;
  int eep2 = 654;
  int kByte = 1024;

  edm::Handle<FEDRawDataCollection> allFedRawData;
  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {
    for ( int iDcc = eem1; iDcc <= eep2; ++iDcc ) {
      int sizeInKB = allFedRawData->FEDData(iDcc).size()/kByte;
      if(iDcc >= eem1 && iDcc <= eem2) nfedEEminus += sizeInKB;
      if(iDcc >= eep1 && iDcc <= eep2) nfedEEplus += sizeInKB;
    }
  }
  else LogWarning("EETrendTask") << FEDRawDataCollection_ << " is not available";

  ecaldqm::shift2Right(nFEDEEminusRawDataMinutely_->getTProfile(), minuteBinDiff);
  nFEDEEminusRawDataMinutely_->Fill(minuteDiff,nfedEEminus);

  ecaldqm::shift2Right(nFEDEEplusRawDataMinutely_->getTProfile(), minuteBinDiff);
  nFEDEEplusRawDataMinutely_->Fill(minuteDiff,nfedEEplus);

  ecaldqm::shift2Right(nFEDEEminusRawDataHourly_->getTProfile(), hourBinDiff);
  nFEDEEminusRawDataHourly_->Fill(hourDiff,nfedEEminus);

  ecaldqm::shift2Right(nFEDEEplusRawDataHourly_->getTProfile(), hourBinDiff);
  nFEDEEplusRawDataHourly_->Fill(hourDiff,nfedEEplus);

}


void EETrendTask::updateTime(){

  last_time_ = current_time_;
  current_time_ = time(NULL);

}


