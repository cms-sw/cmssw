/*
 * \file EBTrendTask.cc
 *
 * $Date: 2009/11/19 18:14:42 $
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

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBTrendTask.h"

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EBTrendTask::EBTrendTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  // parameters...
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  BasicClusterCollection_ = ps.getParameter<edm::InputTag>("BasicClusterCollection");
  SuperClusterCollection_ = ps.getParameter<edm::InputTag>("SuperClusterCollection");
  EBDetIdCollection0_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection0");
  EBDetIdCollection1_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection1");
  EBDetIdCollection2_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection2");
  EBDetIdCollection3_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection3");
  EcalElectronicsIdCollection1_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection1");
  EcalElectronicsIdCollection2_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection2");
  EcalElectronicsIdCollection3_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection3");
  EcalElectronicsIdCollection4_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection4");
  EcalElectronicsIdCollection5_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection5");
  EcalElectronicsIdCollection6_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection6");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  EBSRFlagCollection_ = ps.getParameter<edm::InputTag>("EBSRFlagCollection");

  // histograms...
  nEBDigiMinutely_ = 0;
  nEcalPnDiodeDigiMinutely_ = 0;
  nEcalRecHitMinutely_ = 0;
  nEcalTrigPrimDigiMinutely_ = 0;
  nBasicClusterMinutely_ = 0;
  nBasicClusterSizeMinutely_ = 0;
  nSuperClusterMinutely_ = 0;
  nSuperClusterSizeMinutely_ = 0;
  nIntegrityErrorMinutely_ = 0;
  nFEDEBRawDataMinutely_ = 0;
  nEBSRFlagMinutely_ = 0;

  nEBDigiHourly_ = 0;
  nEcalPnDiodeDigiHourly_ = 0;
  nEcalRecHitHourly_ = 0;
  nEcalTrigPrimDigiHourly_ = 0;
  nBasicClusterHourly_ = 0;
  nBasicClusterSizeHourly_ = 0;
  nSuperClusterHourly_ = 0;
  nSuperClusterSizeHourly_ = 0;
  nIntegrityErrorHourly_ = 0;
  nFEDEBRawDataHourly_ = 0;
  nEBSRFlagHourly_ = 0;
}


EBTrendTask::~EBTrendTask(){
}


void EBTrendTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTrendTask");
    dqmStore_->rmdir(prefixME_ + "/EBTrendTask");
  }

}


void EBTrendTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  start_time_ = time(NULL);

}


void EBTrendTask::endRun(const Run& r, const EventSetup& c) {

}


void EBTrendTask::reset(void) {

  if(nEBDigiMinutely_) nEBDigiMinutely_->Reset();
  if(nEcalPnDiodeDigiMinutely_) nEcalPnDiodeDigiMinutely_->Reset();
  if(nEcalRecHitMinutely_) nEcalRecHitMinutely_->Reset();
  if(nEcalTrigPrimDigiMinutely_) nEcalTrigPrimDigiMinutely_->Reset();
  if(nBasicClusterMinutely_) nBasicClusterMinutely_->Reset();
  if(nBasicClusterSizeMinutely_) nBasicClusterSizeMinutely_->Reset();
  if(nSuperClusterMinutely_) nSuperClusterMinutely_->Reset();
  if(nSuperClusterSizeMinutely_) nSuperClusterSizeMinutely_->Reset();
  if(nIntegrityErrorMinutely_) nIntegrityErrorMinutely_->Reset();
  if(nFEDEBRawDataMinutely_) nFEDEBRawDataMinutely_->Reset();
  if(nEBSRFlagMinutely_) nEBSRFlagMinutely_->Reset();

  if(nEBDigiHourly_) nEBDigiHourly_->Reset();
  if(nEcalPnDiodeDigiHourly_) nEcalPnDiodeDigiHourly_->Reset();
  if(nEcalRecHitHourly_) nEcalRecHitHourly_->Reset();
  if(nEcalTrigPrimDigiHourly_) nEcalTrigPrimDigiHourly_->Reset();
  if(nBasicClusterHourly_) nBasicClusterHourly_->Reset();
  if(nBasicClusterSizeHourly_) nBasicClusterSizeHourly_->Reset();
  if(nSuperClusterHourly_) nSuperClusterHourly_->Reset();
  if(nSuperClusterSizeHourly_) nSuperClusterSizeHourly_->Reset();
  if(nIntegrityErrorHourly_) nIntegrityErrorHourly_->Reset();
  if(nFEDEBRawDataHourly_) nFEDEBRawDataHourly_->Reset();
  if(nEBSRFlagHourly_) nEBSRFlagHourly_->Reset();

}


void EBTrendTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTrendTask");

    // minutely

    sprintf(histo, "AverageNumberOfEBDigiVs5Minutes");
    nEBDigiMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEBDigiMinutely_->setAxisTitle("Minutes", 1);
    nEBDigiMinutely_->setAxisTitle("Average Number of EBDigi / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfEcalPnDiodeDigiVs5Minutes");
    nEcalPnDiodeDigiMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEcalPnDiodeDigiMinutely_->setAxisTitle("Minutes", 1);
    nEcalPnDiodeDigiMinutely_->setAxisTitle("Average Number of EcalPnDiodeDigi / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfEcalRecHitVs5Minutes");
    nEcalRecHitMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEcalRecHitMinutely_->setAxisTitle("Minutes", 1);
    nEcalRecHitMinutely_->setAxisTitle("Average Number of EcalRecHit / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfEcalTrigPrimDigiVs5Minutes");
    nEcalTrigPrimDigiMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEcalTrigPrimDigiMinutely_->setAxisTitle("Minutes", 1);
    nEcalTrigPrimDigiMinutely_->setAxisTitle("Average Number of EcalTrigPrimDigi / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfBasicClusterVs5Minutes");
    nBasicClusterMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nBasicClusterMinutely_->setAxisTitle("Minutes", 1);
    nBasicClusterMinutely_->setAxisTitle("Average Number of BasicClusters / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfBasicClusterSizeVs5Minutes");
    nBasicClusterSizeMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nBasicClusterSizeMinutely_->setAxisTitle("Minutes", 1);
    nBasicClusterSizeMinutely_->setAxisTitle("Average Size of BasicClusters / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfSuperClusterVs5Minutes");
    nSuperClusterMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nSuperClusterMinutely_->setAxisTitle("Minutes", 1);
    nSuperClusterMinutely_->setAxisTitle("Average Number of SuperClusters / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfSuperClusterSizeVs5Minutes");
    nSuperClusterSizeMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nSuperClusterSizeMinutely_->setAxisTitle("Minutes", 1);
    nSuperClusterSizeMinutely_->setAxisTitle("Average Size of SuperClusters / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfIntegrityErrorVs5Minutes");
    nIntegrityErrorMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nIntegrityErrorMinutely_->setAxisTitle("Minutes", 1);
    nIntegrityErrorMinutely_->setAxisTitle("Average IntegrityErrors / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfFEDEBRawDataVs5Minutes");
    nFEDEBRawDataMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nFEDEBRawDataMinutely_->setAxisTitle("Minutes", 1);
    nFEDEBRawDataMinutely_->setAxisTitle("Average Number of FEDRawData in EB / 5 minutes", 2);

    sprintf(histo, "AverageNumberOfEBSRFlagVs5Minutes");
    nEBSRFlagMinutely_ = dqmStore_->bookProfile(histo, histo, 12, 0.0, 60.0, 100, 0.0, 1.0e6, "s");
    nEBSRFlagMinutely_->setAxisTitle("Minutes", 1);
    nEBSRFlagMinutely_->setAxisTitle("Average Number of EBSRFlag / 5 minutes", 2);

    // hourly

    sprintf(histo, "AverageNumberOfEBDigiVs1Hour");
    nEBDigiHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEBDigiHourly_->setAxisTitle("Hours", 1);
    nEBDigiHourly_->setAxisTitle("Average Number of EBDigi / hour", 2);

    sprintf(histo, "AverageNumberOfEcalPnDiodeDigiVs1Hour");
    nEcalPnDiodeDigiHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEcalPnDiodeDigiHourly_->setAxisTitle("Hours", 1);
    nEcalPnDiodeDigiHourly_->setAxisTitle("Average Number of EcalPnDiodeDigi / hour", 2);

    sprintf(histo, "AverageNumberOfEcalRecHitVs1Hour");
    nEcalRecHitHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEcalRecHitHourly_->setAxisTitle("Hours", 1);
    nEcalRecHitHourly_->setAxisTitle("Average Number of EcalRecHit / hour", 2);

    sprintf(histo, "AverageNumberOfEcalTrigPrimDigiVs1Hour");
    nEcalTrigPrimDigiHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEcalTrigPrimDigiHourly_->setAxisTitle("Hours", 1);
    nEcalTrigPrimDigiHourly_->setAxisTitle("Average Number of EcalTrigPrimDigi / hour", 2);

    sprintf(histo, "AverageNumberOfBasicClusterVs1Hour");
    nBasicClusterHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nBasicClusterHourly_->setAxisTitle("Hours", 1);
    nBasicClusterHourly_->setAxisTitle("Average Number of BasicClusters / hour", 2);

    sprintf(histo, "AverageNumberOfBasicClusterSizeVs1Hour");
    nBasicClusterSizeHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nBasicClusterSizeHourly_->setAxisTitle("Hours", 1);
    nBasicClusterSizeHourly_->setAxisTitle("Average Size of BasicClusters / hour", 2);

    sprintf(histo, "AverageNumberOfSuperClusterVs1Hour");
    nSuperClusterHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nSuperClusterHourly_->setAxisTitle("Hours", 1);
    nSuperClusterHourly_->setAxisTitle("Average Number of SuperClusters / hour", 2);

    sprintf(histo, "AverageNumberOfSuperClusterSizeVs1Hour");
    nSuperClusterSizeHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nSuperClusterSizeHourly_->setAxisTitle("Hours", 1);
    nSuperClusterSizeHourly_->setAxisTitle("Average Size of SuperClusters / hour", 2);

    sprintf(histo, "AverageNumberOfIntegrityErrorVs1Hour");
    nIntegrityErrorHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nIntegrityErrorHourly_->setAxisTitle("Hours", 1);
    nIntegrityErrorHourly_->setAxisTitle("Average IntegrityErrors / hour", 2);

    sprintf(histo, "AverageNumberOfFEDEBRawDataVs1Hour");
    nFEDEBRawDataHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nFEDEBRawDataHourly_->setAxisTitle("Hours", 1);
    nFEDEBRawDataHourly_->setAxisTitle("Average Number of FEDRawData in EB / hour", 2);

    sprintf(histo, "AverageNumberOfEBSRFlagVs1Hour");
    nEBSRFlagHourly_ = dqmStore_->bookProfile(histo, histo, 24, 0.0, 24.0, 100, 0.0, 1.0e6, "s");
    nEBSRFlagHourly_->setAxisTitle("Hours", 1);
    nEBSRFlagHourly_->setAxisTitle("Average Number of EBSRFlag / hour", 2);

  }

}


void EBTrendTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTrendTask");

    if(nEBDigiMinutely_) dqmStore_->removeElement( nEBDigiMinutely_->getName());
    nEBDigiMinutely_ = 0;
    if(nEcalPnDiodeDigiMinutely_) dqmStore_->removeElement( nEcalPnDiodeDigiMinutely_->getName());
    nEcalPnDiodeDigiMinutely_ = 0;
    if(nEcalRecHitMinutely_) dqmStore_->removeElement( nEcalRecHitMinutely_->getName());
    nEcalRecHitMinutely_ = 0;
    if(nEcalTrigPrimDigiMinutely_) dqmStore_->removeElement( nEcalTrigPrimDigiMinutely_->getName());
    nEcalTrigPrimDigiMinutely_ = 0;
    if(nBasicClusterMinutely_) dqmStore_->removeElement( nBasicClusterMinutely_->getName());
    nBasicClusterMinutely_ = 0;
    if(nBasicClusterSizeMinutely_) dqmStore_->removeElement( nBasicClusterSizeMinutely_->getName());
    nBasicClusterSizeMinutely_ = 0;
    if(nSuperClusterMinutely_) dqmStore_->removeElement( nSuperClusterMinutely_->getName());
    nSuperClusterMinutely_ = 0;
    if(nSuperClusterSizeMinutely_) dqmStore_->removeElement( nSuperClusterSizeMinutely_->getName());
    nSuperClusterSizeMinutely_ = 0;
    if(nIntegrityErrorMinutely_) dqmStore_->removeElement( nIntegrityErrorMinutely_->getName());
    nIntegrityErrorMinutely_ = 0;
    if(nFEDEBRawDataMinutely_) dqmStore_->removeElement( nFEDEBRawDataMinutely_->getName());
    nFEDEBRawDataMinutely_ = 0;
    if(nEBSRFlagMinutely_) dqmStore_->removeElement( nEBSRFlagMinutely_->getName());
    nEBSRFlagMinutely_ =0;

    if(nEBDigiHourly_) dqmStore_->removeElement( nEBDigiHourly_->getName());
    nEBDigiHourly_ = 0;
    if(nEcalPnDiodeDigiHourly_) dqmStore_->removeElement( nEcalPnDiodeDigiHourly_->getName());
    nEcalPnDiodeDigiHourly_ = 0;
    if(nEcalRecHitHourly_) dqmStore_->removeElement( nEcalRecHitHourly_->getName());
    nEcalRecHitHourly_ = 0;
    if(nEcalTrigPrimDigiHourly_) dqmStore_->removeElement( nEcalTrigPrimDigiHourly_->getName());
    nEcalTrigPrimDigiHourly_ = 0;
    if(nBasicClusterHourly_) dqmStore_->removeElement( nBasicClusterHourly_->getName());
    nBasicClusterHourly_ = 0;
    if(nBasicClusterSizeHourly_) dqmStore_->removeElement( nBasicClusterSizeHourly_->getName());
    nBasicClusterSizeHourly_ = 0;
    if(nSuperClusterHourly_) dqmStore_->removeElement( nSuperClusterHourly_->getName());
    nSuperClusterHourly_ = 0;
    if(nSuperClusterSizeHourly_) dqmStore_->removeElement( nSuperClusterSizeHourly_->getName());
    nSuperClusterSizeHourly_ = 0;
    if(nIntegrityErrorHourly_) dqmStore_->removeElement( nIntegrityErrorHourly_->getName());
    nIntegrityErrorHourly_ = 0;
    if(nFEDEBRawDataHourly_) dqmStore_->removeElement( nFEDEBRawDataHourly_->getName());
    nFEDEBRawDataHourly_ = 0;
    if(nEBSRFlagHourly_) dqmStore_->removeElement( nEBSRFlagHourly_->getName());
    nEBSRFlagHourly_ = 0;

  }

  init_ = false;

}


void EBTrendTask::endJob(void){

  LogInfo("EBTrendTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}


void EBTrendTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --------------------------------------------------
  // Collect time information
  // --------------------------------------------------

  updateTime();

  long int diff_current_start = current_time_ - start_time_;
  long int diff_last_start = last_time_ - start_time_;
  LogInfo("EBTrendTask") << "time difference is negative in " << ievt_ << " events\n"
			 << "\tcurrent - start time = " << diff_current_start
			 << ", \tlast - start time = " << diff_last_start << endl;


  // --------------------------------------------------
  // Calculate time interval and bin width
  // --------------------------------------------------

  //  int minuteBinWidth = int(nBasicClusterMinutely_->getTProfile()->GetXaxis()->GetBinWidth(1));
  int minuteBinWidth = 5;
  long int minuteBinDiff = diff_current_start/60/minuteBinWidth - diff_last_start/60/minuteBinWidth;
  long int minuteDiff = (current_time_ - last_time_)/60;

  //  int hourBinWidth = int(nBasicClusterHourly_->getTProfile()->GetXaxis()->GetBinWidth(1));
  int hourBinWidth = 1;
  long int hourBinDiff = diff_current_start/3600/hourBinWidth - diff_last_start/3600/hourBinWidth;
  long int hourDiff = (current_time_ - last_time_)/3600;

  if(minuteDiff >= minuteBinWidth) {
    while(minuteDiff >= minuteBinWidth) minuteDiff -= minuteBinWidth;
  }
  if(hourDiff >= hourBinWidth){
    while(hourDiff >= hourBinWidth) hourDiff -= hourBinWidth;
  }


  // --------------------------------------------------
  // EBDigiCollection
  // --------------------------------------------------
  int ndc = 0;
  Handle<EBDigiCollection> digis;
  if ( e.getByLabel(EBDigiCollection_, digis) ) ndc = digis->size();
  else LogWarning("EBTrendTask") << EBDigiCollection_ << " is not available";

  shift2Right(nEBDigiMinutely_->getTProfile(), minuteBinDiff);
  nEBDigiMinutely_->Fill(minuteDiff,ndc);
  
  shift2Right(nEBDigiHourly_->getTProfile(), hourBinDiff);
  nEBDigiHourly_->Fill(hourDiff,ndc);


  // --------------------------------------------------
  // EcalPnDiodeDigiCollection
  // --------------------------------------------------
  int npdc = 0;
  Handle<EcalPnDiodeDigiCollection> pns;
  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) npdc = pns->size();
  else LogWarning("EBTrendTask") << EcalPnDiodeDigiCollection_ << " is not available";

  shift2Right(nEcalPnDiodeDigiMinutely_->getTProfile(), minuteBinDiff);
  nEcalPnDiodeDigiMinutely_->Fill(minuteDiff,npdc);
  
  shift2Right(nEcalPnDiodeDigiHourly_->getTProfile(), hourBinDiff);
  nEcalPnDiodeDigiHourly_->Fill(hourDiff,npdc);


  // --------------------------------------------------
  // EcalRecHitCollection
  // --------------------------------------------------
  int nrhc = 0;
  Handle<EcalRecHitCollection> hits;
  if ( e.getByLabel(EcalRecHitCollection_, hits) ) nrhc = hits->size();
  else LogWarning("EBTrendTask") << EcalRecHitCollection_ << " is not available";

  shift2Right(nEcalRecHitMinutely_->getTProfile(), minuteBinDiff);
  nEcalRecHitMinutely_->Fill(minuteDiff,nrhc);
  
  shift2Right(nEcalRecHitHourly_->getTProfile(), hourBinDiff);
  nEcalRecHitHourly_->Fill(hourDiff,nrhc);

  // --------------------------------------------------
  // EcalTrigPrimDigiCollection
  // --------------------------------------------------
  int ntpdc = 0;
  Handle<EcalTrigPrimDigiCollection> tpdigis;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, tpdigis) ) ntpdc = tpdigis->size();
  else LogWarning("EBTrendTask") << EcalTrigPrimDigiCollection_ << " is not available";

  shift2Right(nEcalTrigPrimDigiMinutely_->getTProfile(), minuteBinDiff);
  nEcalTrigPrimDigiMinutely_->Fill(minuteDiff,ntpdc);
  
  shift2Right(nEcalTrigPrimDigiHourly_->getTProfile(), hourBinDiff);
  nEcalTrigPrimDigiHourly_->Fill(hourDiff,ntpdc);

  // --------------------------------------------------
  // BasicClusters
  // --------------------------------------------------
  int nbcc = 0;
  float nbcc_size = 0.0;
  Handle<BasicClusterCollection> pBasicClusters;
  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {
    nbcc = pBasicClusters->size();
    for(reco::BasicClusterCollection::const_iterator it = pBasicClusters->begin();
	it != pBasicClusters->end(); it++){
      nbcc_size += it->size();
    }
    if(nbcc == 0) nbcc_size = 0;
    else nbcc_size = nbcc_size / nbcc;
  }
  else LogWarning("EBTrendTask") << BasicClusterCollection_ << " is not available";

  shift2Right(nBasicClusterMinutely_->getTProfile(), minuteBinDiff);
  nBasicClusterMinutely_->Fill(minuteDiff,nbcc);
  
  shift2Right(nBasicClusterHourly_->getTProfile(), hourBinDiff);
  nBasicClusterHourly_->Fill(hourDiff,nbcc);

  shift2Right(nBasicClusterSizeMinutely_->getTProfile(), minuteBinDiff);
  nBasicClusterSizeMinutely_->Fill(minuteDiff,nbcc);
  
  shift2Right(nBasicClusterSizeHourly_->getTProfile(), hourBinDiff);
  nBasicClusterSizeHourly_->Fill(hourDiff,nbcc);

  // --------------------------------------------------
  // SuperClusters
  // --------------------------------------------------
  int nscc = 0;
  float nscc_size = 0.0;
  Handle<SuperClusterCollection> pSuperClusters;
  if ( e.getByLabel(SuperClusterCollection_, pSuperClusters) ) {
    nscc = pSuperClusters->size();
    for(reco::SuperClusterCollection::const_iterator it = pSuperClusters->begin();
	it != pSuperClusters->end(); it++){
      nscc_size += it->clustersSize();
    }
    if(nscc == 0) nscc_size = 0;
    else nscc_size = nscc_size / nscc;
  }
  else LogWarning("EBTrendTask") << SuperClusterCollection_ << " is not available";

  shift2Right(nSuperClusterMinutely_->getTProfile(), minuteBinDiff);
  nSuperClusterMinutely_->Fill(minuteDiff,nscc);
  
  shift2Right(nSuperClusterHourly_->getTProfile(), hourBinDiff);
  nSuperClusterHourly_->Fill(hourDiff,nscc);

  shift2Right(nSuperClusterSizeMinutely_->getTProfile(), minuteBinDiff);
  nSuperClusterSizeMinutely_->Fill(minuteDiff,nscc);
  
  shift2Right(nSuperClusterSizeHourly_->getTProfile(), hourBinDiff);
  nSuperClusterSizeHourly_->Fill(hourDiff,nscc);


  // --------------------------------------------------
  // Integrity errors (sum of collections' sizes)
  // --------------------------------------------------
  //  double errorSum = 0.0;

  // --------------------------------------------------
  // EBDetIdCollection0
  // --------------------------------------------------
  int ndic0 = 0;
  Handle<EBDetIdCollection> ids0;
  if ( e.getByLabel(EBDetIdCollection0_, ids0) ) ndic0 = ids0->size();
  else LogWarning("EBTrendTask") << EBDetIdCollection0_ << " is not available";


  // --------------------------------------------------
  // EBDetIdCollection1
  // --------------------------------------------------
  int ndic1 = 0;
  Handle<EBDetIdCollection> ids1;
  if ( e.getByLabel(EBDetIdCollection1_, ids1) ) ndic1 = ids1->size();
  else LogWarning("EBTrendTask") << EBDetIdCollection1_ << " is not available";


  // --------------------------------------------------
  // EBDetIdCollection2
  // --------------------------------------------------
  int ndic2 = 0;
  Handle<EBDetIdCollection> ids2;
  if ( e.getByLabel(EBDetIdCollection2_, ids2) ) ndic2 = ids2->size();
  else LogWarning("EBTrendTask") << EBDetIdCollection2_ << " is not available";


  // --------------------------------------------------
  // EBDetIdCollection3
  // --------------------------------------------------
  int ndic3 = 0;
  Handle<EBDetIdCollection> ids3;
  if ( e.getByLabel(EBDetIdCollection3_, ids3) ) ndic3 = ids3->size();
  else LogWarning("EBTrendTask") << EBDetIdCollection3_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection1
  // --------------------------------------------------
  int neic1 = 0;
  Handle<EcalElectronicsIdCollection> eids1;
  if ( e.getByLabel(EcalElectronicsIdCollection1_, eids1) ) neic1 = eids1->size();
  else LogWarning("EBTrendTask") << EcalElectronicsIdCollection1_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection2
  // --------------------------------------------------
  int neic2 = 0;
  Handle<EcalElectronicsIdCollection> eids2;
  if ( e.getByLabel(EcalElectronicsIdCollection2_, eids2) ) neic2 = eids2->size();
  else LogWarning("EBTrendTask") << EcalElectronicsIdCollection2_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection3
  // --------------------------------------------------
  int neic3 = 0;
  Handle<EcalElectronicsIdCollection> eids3;
  if ( e.getByLabel(EcalElectronicsIdCollection3_, eids3) ) neic3 = eids3->size();
  else LogWarning("EBTrendTask") << EcalElectronicsIdCollection3_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection4
  // --------------------------------------------------
  int neic4 = 0;
  Handle<EcalElectronicsIdCollection> eids4;
  if ( e.getByLabel(EcalElectronicsIdCollection4_, eids4) ) neic4 = eids4->size();
  else LogWarning("EBTrendTask") << EcalElectronicsIdCollection4_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection5
  // --------------------------------------------------
  int neic5 = 0;
  Handle<EcalElectronicsIdCollection> eids5;
  if ( e.getByLabel(EcalElectronicsIdCollection5_, eids5) ) neic5 = eids5->size();
  else LogWarning("EBTrendTask") << EcalElectronicsIdCollection5_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection6
  // --------------------------------------------------
  int neic6 = 0;
  Handle<EcalElectronicsIdCollection> eids6;
  if ( e.getByLabel(EcalElectronicsIdCollection6_, eids6) ) neic6 = eids6->size();
  else LogWarning("EBTrendTask") << EcalElectronicsIdCollection6_ << " is not available";


  // --------------------------------------------------
  // Integrity errors (sum of collections' sizes)
  // --------------------------------------------------
  double errorSum = ndic0 + ndic1 + ndic2 + ndic3 +
    neic1 + neic2 + neic3 + neic4 + neic5 + neic6;

  shift2Right(nIntegrityErrorMinutely_->getTProfile(), minuteBinDiff);
  nIntegrityErrorMinutely_->Fill(minuteDiff,errorSum);
  
  shift2Right(nIntegrityErrorHourly_->getTProfile(), hourBinDiff);
  nIntegrityErrorHourly_->Fill(hourDiff,errorSum);

  // --------------------------------------------------
  // FEDRawDataCollection
  // --------------------------------------------------
  int nfedEB      = 0;

  // Barrel FEDs : 610 - 645
  // Endcap FEDs : 601-609 (EE-) and 646-654 (EE+) 
  int eb1 = 610;
  int eb2 = 645;
  int kByte = 1024;

  edm::Handle<FEDRawDataCollection> allFedRawData;
  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {
    for ( int iDcc = eb1; iDcc <= eb2; ++iDcc ) {
      int sizeInKB = allFedRawData->FEDData(iDcc).size()/kByte;
      if(iDcc >= eb1  && iDcc <= eb2)  nfedEB += sizeInKB;
    }
  }
  else LogWarning("EBTrendTask") << FEDRawDataCollection_ << " is not available";

  shift2Right(nFEDEBRawDataMinutely_->getTProfile(), minuteBinDiff);
  nFEDEBRawDataMinutely_->Fill(minuteDiff,nfedEB);

  shift2Right(nFEDEBRawDataHourly_->getTProfile(), hourBinDiff);
  nFEDEBRawDataHourly_->Fill(hourDiff,nfedEB);


  // --------------------------------------------------
  // EBSRFlagCollection
  // --------------------------------------------------
  int nsfc = 0;
  Handle<EBSrFlagCollection> ebSrFlags;
  if ( e.getByLabel(EBSRFlagCollection_,ebSrFlags) ) nsfc = ebSrFlags->size();
  else LogWarning("EBTrendTask") << EBSRFlagCollection_ << " is not available";

  shift2Right(nEBSRFlagMinutely_->getTProfile(), minuteBinDiff);
  nEBSRFlagMinutely_->Fill(minuteDiff,nsfc);
  
  shift2Right(nEBSRFlagHourly_->getTProfile(), hourBinDiff);
  nEBSRFlagHourly_->Fill(hourDiff,nsfc);


//   printf("run(%d), event(%d), ndc(%d), npdc(%d), nrhc(%d), ntpdc(%d), nbcc(%d), ",
// 	 e.id().run(),e.id().event(), ndc, npdc, nrhc, ntpdc, nbcc);
//   printf("nscc(%d), ndic0(%d), ndic1(%d), ndic2(%d), ndic3(%d), neic1(%d), neic2(%d), neic3(%d), ",
// 	 nscc, ndic0, ndic1, ndic2, ndic3, neic1, neic2, neic3);
//   printf("neic4(%d), neic5(%d), neic6(%d), errorSum(%f), nsfc(%d), ",
// 	 neic4, neic5, neic6, errorSum, nsfc);
//   printf("nfedEB(%d), nfedEEminus(%d), nfedEEplus(%d)\n",
// 	 nfedEB, nfedEEminus, nfedEEplus);

}


void EBTrendTask::updateTime(){

  last_time_ = current_time_;
  current_time_ = time(NULL);

}


void EBTrendTask::shift2Right(TProfile* p, int bins){

  if(bins <= 0) return;

  if(!p->GetSumw2()) p->Sumw2();
  int nBins = p->GetXaxis()->GetNbins();

  // by shifting n bin to the right, the number of entries are
  // reduced by the number in n bins including the overflow bin.
  double nentries = p->GetEntries();
  for(int i=0; i<bins; i++) nentries -= p->GetBinEntries(i);
  p->SetEntries(nentries);
  
  // the last bin goes to overflow
  // each bin moves to the right

  TArrayD* sumw2 = p->GetSumw2();

  for(int i=nBins+1; i>bins; i--) {
    // GetBinContent return binContent/binEntries
    p->SetBinContent(i, p->GetBinContent(i-bins)*p->GetBinEntries(i-bins));
    p->SetBinEntries(i,p->GetBinEntries(i-bins));
    sumw2->SetAt(sumw2->GetAt(i-bins),i);
  }

}


void EBTrendTask::shift2Left(TProfile* p, int bins){

  if(bins <= 0) return;

  if(!p->GetSumw2()) p->Sumw2();
  int nBins = p->GetXaxis()->GetNbins();

  // by shifting n bin to the left, the number of entries are
  // reduced by the number in n bins including the underflow bin.
  double nentries = p->GetEntries();
  for(int i=0; i<bins; i++) nentries -= p->GetBinEntries(i);
  p->SetEntries(nentries);
  
  // the first bin goes to underflow
  // each bin moves to the right

  TArrayD* sumw2 = p->GetSumw2();

  for(int i=0; i<=nBins+1-bins; i++) {
    // GetBinContent return binContent/binEntries
    p->SetBinContent(i, p->GetBinContent(i+bins)*p->GetBinEntries(i+bins));
    p->SetBinEntries(i,p->GetBinEntries(i+bins));
    sumw2->SetAt(sumw2->GetAt(i+bins),i);
  }

}
