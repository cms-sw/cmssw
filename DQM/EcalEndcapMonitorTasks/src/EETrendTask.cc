/*
 * \file EETrendTask.cc
 *
 * $Date: 2011/08/30 09:28:42 $
 * $Revision: 1.13 $
 * \author Dongwook Jang, Soon Yung Jun
 *
*/

#include <iostream>
#include <fstream>
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

#include "TProfile.h"

EETrendTask::EETrendTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // parameters...
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  BasicClusterCollection_ = ps.getParameter<edm::InputTag>("BasicClusterCollection");
  SuperClusterCollection_ = ps.getParameter<edm::InputTag>("SuperClusterCollection");
  EEDetIdCollection0_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection0");
  EEDetIdCollection1_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection1");
  EEDetIdCollection2_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection2");
  EEDetIdCollection3_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection3");
  EcalElectronicsIdCollection1_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection1");
  EcalElectronicsIdCollection2_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection2");
  EcalElectronicsIdCollection3_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection3");
  EcalElectronicsIdCollection4_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection4");
  EcalElectronicsIdCollection5_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection5");
  EcalElectronicsIdCollection6_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection6");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  EESRFlagCollection_ = ps.getParameter<edm::InputTag>("EESRFlagCollection");

  // histograms...
  nEEDigiMinutely_ = 0;
  nEcalPnDiodeDigiMinutely_ = 0;
  nEcalRecHitMinutely_ = 0;
  nEcalTrigPrimDigiMinutely_ = 0;
  nBasicClusterMinutely_ = 0;
  nBasicClusterSizeMinutely_ = 0;
  nSuperClusterMinutely_ = 0;
  nSuperClusterSizeMinutely_ = 0;
  nIntegrityErrorMinutely_ = 0;
  nFEDEERawDataMinutely_ = 0;
  nEESRFlagMinutely_ = 0;

  nEEDigiHourly_ = 0;
  nEcalPnDiodeDigiHourly_ = 0;
  nEcalRecHitHourly_ = 0;
  nEcalTrigPrimDigiHourly_ = 0;
  nBasicClusterHourly_ = 0;
  nBasicClusterSizeHourly_ = 0;
  nSuperClusterHourly_ = 0;
  nSuperClusterSizeHourly_ = 0;
  nIntegrityErrorHourly_ = 0;
  nFEDEERawDataHourly_ = 0;
  nEESRFlagHourly_ = 0;

  current_time_ = 0;
  start_time_ = 0;
  last_time_ = 0;

  ievt_ = 0;
}


EETrendTask::~EETrendTask(){
}


void EETrendTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");
    dqmStore_->rmdir(prefixME_ + "/Trend");
  }

}


void EETrendTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  start_time_ = time(NULL);

}


void EETrendTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}


void EETrendTask::reset(void) {

  if(nEEDigiMinutely_) nEEDigiMinutely_->Reset();
  if(nEcalPnDiodeDigiMinutely_) nEcalPnDiodeDigiMinutely_->Reset();
  if(nEcalRecHitMinutely_) nEcalRecHitMinutely_->Reset();
  if(nEcalTrigPrimDigiMinutely_) nEcalTrigPrimDigiMinutely_->Reset();
  if(nBasicClusterMinutely_) nBasicClusterMinutely_->Reset();
  if(nBasicClusterSizeMinutely_) nBasicClusterSizeMinutely_->Reset();
  if(nSuperClusterMinutely_) nSuperClusterMinutely_->Reset();
  if(nSuperClusterSizeMinutely_) nSuperClusterSizeMinutely_->Reset();
  if(nIntegrityErrorMinutely_) nIntegrityErrorMinutely_->Reset();
  if(nFEDEERawDataMinutely_) nFEDEERawDataMinutely_->Reset();
  if(nEESRFlagMinutely_) nEESRFlagMinutely_->Reset();

  if(nEEDigiHourly_) nEEDigiHourly_->Reset();
  if(nEcalPnDiodeDigiHourly_) nEcalPnDiodeDigiHourly_->Reset();
  if(nEcalRecHitHourly_) nEcalRecHitHourly_->Reset();
  if(nEcalTrigPrimDigiHourly_) nEcalTrigPrimDigiHourly_->Reset();
  if(nBasicClusterHourly_) nBasicClusterHourly_->Reset();
  if(nBasicClusterSizeHourly_) nBasicClusterSizeHourly_->Reset();
  if(nSuperClusterHourly_) nSuperClusterHourly_->Reset();
  if(nSuperClusterSizeHourly_) nSuperClusterSizeHourly_->Reset();
  if(nIntegrityErrorHourly_) nIntegrityErrorHourly_->Reset();
  if(nFEDEERawDataHourly_) nFEDEERawDataHourly_->Reset();
  if(nEESRFlagHourly_) nEESRFlagHourly_->Reset();

}


void EETrendTask::setup(void){

  init_ = true;

  std::string name;
  std::string binning;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");

    // minutely
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend/ShortTerm");

    binning = "5 min bin EE";

    name = "TrendTask num digis " + binning;
    nEEDigiMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEEDigiMinutely_->setAxisTitle("Minutes", 1);
    nEEDigiMinutely_->setAxisTitle("Average Number of EEDigi / 5 minutes", 2);

    name = "TrendTask num PN digis " + binning;
    nEcalPnDiodeDigiMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEcalPnDiodeDigiMinutely_->setAxisTitle("Minutes", 1);
    nEcalPnDiodeDigiMinutely_->setAxisTitle("Average Number of EcalPnDiodeDigi / 5 minutes", 2);

    name = "TrendTask num rec hits " + binning;
    nEcalRecHitMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEcalRecHitMinutely_->setAxisTitle("Minutes", 1);
    nEcalRecHitMinutely_->setAxisTitle("Average Number of EcalRecHit / 5 minutes", 2);

    name = "TrendTask num TP digis " + binning;
    nEcalTrigPrimDigiMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEcalTrigPrimDigiMinutely_->setAxisTitle("Minutes", 1);
    nEcalTrigPrimDigiMinutely_->setAxisTitle("Average Number of EcalTrigPrimDigi / 5 minutes", 2);

    name = "TrendTask num BCs " + binning;
    nBasicClusterMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nBasicClusterMinutely_->setAxisTitle("Minutes", 1);
    nBasicClusterMinutely_->setAxisTitle("Average Number of BasicClusters / 5 minutes", 2);

    name = "TrendTask BC size " + binning;
    nBasicClusterSizeMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nBasicClusterSizeMinutely_->setAxisTitle("Minutes", 1);
    nBasicClusterSizeMinutely_->setAxisTitle("Average Size of BasicClusters / 5 minutes", 2);

    name = "TrendTask num SCs " + binning;
    nSuperClusterMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nSuperClusterMinutely_->setAxisTitle("Minutes", 1);
    nSuperClusterMinutely_->setAxisTitle("Average Number of SuperClusters / 5 minutes", 2);

    name = "TrendTask SC size " + binning;
    nSuperClusterSizeMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nSuperClusterSizeMinutely_->setAxisTitle("Minutes", 1);
    nSuperClusterSizeMinutely_->setAxisTitle("Average Size of SuperClusters / 5 minutes", 2);

    name = "TrendTask num integrity errors " + binning;
    nIntegrityErrorMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nIntegrityErrorMinutely_->setAxisTitle("Minutes", 1);
    nIntegrityErrorMinutely_->setAxisTitle("Average IntegrityErrors / 5 minutes", 2);

    name = "TrendTask DCC event size " + binning;
    nFEDEERawDataMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nFEDEERawDataMinutely_->setAxisTitle("Minutes", 1);
    nFEDEERawDataMinutely_->setAxisTitle("Average Number of FEDRawData in EE / 5 minutes", 2);

    name = "TrendTask num SR flags " + binning;
    nEESRFlagMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEESRFlagMinutely_->setAxisTitle("Minutes", 1);
    nEESRFlagMinutely_->setAxisTitle("Average Number of EESRFlag / 5 minutes", 2);


    // hourly
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend/LongTerm");

    binning = "20 min bin EB";

    name = "TrendTask num digis " + binning;
    nEEDigiHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEEDigiHourly_->setAxisTitle("Hours", 1);
    nEEDigiHourly_->setAxisTitle("Average Number of EEDigi / 20 minutes", 2);

    name = "TrendTask num PN digis " + binning;
    nEcalPnDiodeDigiHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEcalPnDiodeDigiHourly_->setAxisTitle("Hours", 1);
    nEcalPnDiodeDigiHourly_->setAxisTitle("Average Number of EcalPnDiodeDigi / 20 minutes", 2);

    name = "TrendTask num rec hits " + binning;
    nEcalRecHitHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEcalRecHitHourly_->setAxisTitle("Hours", 1);
    nEcalRecHitHourly_->setAxisTitle("Average Number of EcalRecHit / 20 minutes", 2);

    name = "TrendTask num TP digis " + binning;
    nEcalTrigPrimDigiHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEcalTrigPrimDigiHourly_->setAxisTitle("Hours", 1);
    nEcalTrigPrimDigiHourly_->setAxisTitle("Average Number of EcalTrigPrimDigi / 20 minutes", 2);

    name = "TrendTask num BCs " + binning;
    nBasicClusterHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nBasicClusterHourly_->setAxisTitle("Hours", 1);
    nBasicClusterHourly_->setAxisTitle("Average Number of BasicClusters / 20 minutes", 2);

    name = "TrendTask BC size " + binning;
    nBasicClusterSizeHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nBasicClusterSizeHourly_->setAxisTitle("Hours", 1);
    nBasicClusterSizeHourly_->setAxisTitle("Average Size of BasicClusters / 20 minutes", 2);

    name = "TrendTask num SCs " + binning;
    nSuperClusterHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nSuperClusterHourly_->setAxisTitle("Hours", 1);
    nSuperClusterHourly_->setAxisTitle("Average Number of SuperClusters / 20 minutes", 2);

    name = "TrendTask SC size " + binning;
    nSuperClusterSizeHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nSuperClusterSizeHourly_->setAxisTitle("Hours", 1);
    nSuperClusterSizeHourly_->setAxisTitle("Average Size of SuperClusters / 20 minutes", 2);

    name = "TrendTask num integrity errors " + binning;
    nIntegrityErrorHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nIntegrityErrorHourly_->setAxisTitle("Hours", 1);
    nIntegrityErrorHourly_->setAxisTitle("Average IntegrityErrors / 20 minutes", 2);

    name = "TrendTask DCC event size " + binning;
    nFEDEERawDataHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nFEDEERawDataHourly_->setAxisTitle("Hours", 1);
    nFEDEERawDataHourly_->setAxisTitle("Average Number of FEDRawData in EE / 20 minutes", 2);

    name = "TrendTask num SR flags " + binning;
    nEESRFlagHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEESRFlagHourly_->setAxisTitle("Hours", 1);
    nEESRFlagHourly_->setAxisTitle("Average Number of EESRFlag / 20 minutes", 2);

  }

}


void EETrendTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");

    if(nEEDigiMinutely_) dqmStore_->removeElement( nEEDigiMinutely_->getFullname());
    nEEDigiMinutely_ = 0;
    if(nEcalPnDiodeDigiMinutely_) dqmStore_->removeElement( nEcalPnDiodeDigiMinutely_->getFullname());
    nEcalPnDiodeDigiMinutely_ = 0;
    if(nEcalRecHitMinutely_) dqmStore_->removeElement( nEcalRecHitMinutely_->getFullname());
    nEcalRecHitMinutely_ = 0;
    if(nEcalTrigPrimDigiMinutely_) dqmStore_->removeElement( nEcalTrigPrimDigiMinutely_->getFullname());
    nEcalTrigPrimDigiMinutely_ = 0;
    if(nBasicClusterMinutely_) dqmStore_->removeElement( nBasicClusterMinutely_->getFullname());
    nBasicClusterMinutely_ = 0;
    if(nBasicClusterSizeMinutely_) dqmStore_->removeElement( nBasicClusterSizeMinutely_->getFullname());
    nBasicClusterSizeMinutely_ = 0;
    if(nSuperClusterMinutely_) dqmStore_->removeElement( nSuperClusterMinutely_->getFullname());
    nSuperClusterMinutely_ = 0;
    if(nSuperClusterSizeMinutely_) dqmStore_->removeElement( nSuperClusterSizeMinutely_->getFullname());
    nSuperClusterSizeMinutely_ = 0;
    if(nIntegrityErrorMinutely_) dqmStore_->removeElement( nIntegrityErrorMinutely_->getFullname());
    nIntegrityErrorMinutely_ = 0;
    if(nFEDEERawDataMinutely_) dqmStore_->removeElement( nFEDEERawDataMinutely_->getFullname());
    nFEDEERawDataMinutely_ = 0;
    if(nEESRFlagMinutely_) dqmStore_->removeElement( nEESRFlagMinutely_->getFullname());
    nEESRFlagMinutely_ = 0;

    if(nEEDigiHourly_) dqmStore_->removeElement( nEEDigiHourly_->getFullname());
    nEEDigiHourly_ = 0;
    if(nEcalPnDiodeDigiHourly_) dqmStore_->removeElement( nEcalPnDiodeDigiHourly_->getFullname());
    nEcalPnDiodeDigiHourly_ = 0;
    if(nEcalRecHitHourly_) dqmStore_->removeElement( nEcalRecHitHourly_->getFullname());
    nEcalRecHitHourly_ = 0;
    if(nEcalTrigPrimDigiHourly_) dqmStore_->removeElement( nEcalTrigPrimDigiHourly_->getFullname());
    nEcalTrigPrimDigiHourly_ = 0;
    if(nBasicClusterHourly_) dqmStore_->removeElement( nBasicClusterHourly_->getFullname());
    nBasicClusterHourly_ = 0;
    if(nBasicClusterSizeHourly_) dqmStore_->removeElement( nBasicClusterSizeHourly_->getFullname());
    nBasicClusterSizeHourly_ = 0;
    if(nSuperClusterHourly_) dqmStore_->removeElement( nSuperClusterHourly_->getFullname());
    nSuperClusterHourly_ = 0;
    if(nSuperClusterSizeHourly_) dqmStore_->removeElement( nSuperClusterSizeHourly_->getFullname());
    nSuperClusterSizeHourly_ = 0;
    if(nIntegrityErrorHourly_) dqmStore_->removeElement( nIntegrityErrorHourly_->getFullname());
    nIntegrityErrorHourly_ = 0;
    if(nFEDEERawDataHourly_) dqmStore_->removeElement( nFEDEERawDataHourly_->getFullname());
    nFEDEERawDataHourly_ = 0;
    if(nEESRFlagHourly_) dqmStore_->removeElement( nEESRFlagHourly_->getFullname());
    nEESRFlagHourly_ = 0;

  }

  init_ = false;

}


void EETrendTask::endJob(void){

  edm::LogInfo("EETrendTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}


void EETrendTask::analyze(const edm::Event& e, const edm::EventSetup& c){

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
  ecaldqm::calcBins(20,60,start_time_,last_time_,current_time_,hourBinDiff,hourDiff);


  // --------------------------------------------------
  // EEDigiCollection
  // --------------------------------------------------
  int ndc = 0;
  edm::Handle<EEDigiCollection> digis;
  if ( e.getByLabel(EEDigiCollection_, digis) ) ndc = digis->size();
  else edm::LogWarning("EETrendTask") << EEDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEEDigiMinutely_->getTProfile(), minuteBinDiff);
  nEEDigiMinutely_->Fill(minuteDiff,ndc);

  ecaldqm::shift2Right(nEEDigiHourly_->getTProfile(), hourBinDiff);
  nEEDigiHourly_->Fill(hourDiff,ndc);


  // --------------------------------------------------
  // EcalPnDiodeDigiCollection
  // --------------------------------------------------
  int npdc = 0;
  edm::Handle<EcalPnDiodeDigiCollection> pns;
  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) npdc = pns->size();
  else edm::LogWarning("EETrendTask") << EcalPnDiodeDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalPnDiodeDigiMinutely_->getTProfile(), minuteBinDiff);
  nEcalPnDiodeDigiMinutely_->Fill(minuteDiff,npdc);

  ecaldqm::shift2Right(nEcalPnDiodeDigiHourly_->getTProfile(), hourBinDiff);
  nEcalPnDiodeDigiHourly_->Fill(hourDiff,npdc);


  // --------------------------------------------------
  // EcalRecHitCollection
  // --------------------------------------------------
  int nrhc = 0;
  edm::Handle<EcalRecHitCollection> hits;
  if ( e.getByLabel(EcalRecHitCollection_, hits) ) nrhc = hits->size();
  else edm::LogWarning("EETrendTask") << EcalRecHitCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalRecHitMinutely_->getTProfile(), minuteBinDiff);
  nEcalRecHitMinutely_->Fill(minuteDiff,nrhc);

  ecaldqm::shift2Right(nEcalRecHitHourly_->getTProfile(), hourBinDiff);
  nEcalRecHitHourly_->Fill(hourDiff,nrhc);


  // --------------------------------------------------
  // EcalTrigPrimDigiCollection
  // --------------------------------------------------
  int ntpdc = 0;
  edm::Handle<EcalTrigPrimDigiCollection> tpdigis;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, tpdigis) ) ntpdc = tpdigis->size();
  else edm::LogWarning("EETrendTask") << EcalTrigPrimDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalTrigPrimDigiMinutely_->getTProfile(), minuteBinDiff);
  nEcalTrigPrimDigiMinutely_->Fill(minuteDiff,ntpdc);

  ecaldqm::shift2Right(nEcalTrigPrimDigiHourly_->getTProfile(), hourBinDiff);
  nEcalTrigPrimDigiHourly_->Fill(hourDiff,ntpdc);


  // --------------------------------------------------
  // BasicClusters
  // --------------------------------------------------
  int nbcc = 0;
  float nbcc_size = 0.0;
  edm::Handle<reco::BasicClusterCollection> pBasicClusters;
  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {
    nbcc = pBasicClusters->size();
    for(reco::BasicClusterCollection::const_iterator it = pBasicClusters->begin();
	it != pBasicClusters->end(); it++){
      nbcc_size += it->size();
    }
    if(nbcc == 0) nbcc_size = 0;
    else nbcc_size = nbcc_size / nbcc;
  }
  else edm::LogWarning("EETrendTask") << BasicClusterCollection_ << " is not available";

  ecaldqm::shift2Right(nBasicClusterMinutely_->getTProfile(), minuteBinDiff);
  nBasicClusterMinutely_->Fill(minuteDiff,nbcc);

  ecaldqm::shift2Right(nBasicClusterHourly_->getTProfile(), hourBinDiff);
  nBasicClusterHourly_->Fill(hourDiff,nbcc);

  ecaldqm::shift2Right(nBasicClusterSizeMinutely_->getTProfile(), minuteBinDiff);
  nBasicClusterSizeMinutely_->Fill(minuteDiff,nbcc);

  ecaldqm::shift2Right(nBasicClusterSizeHourly_->getTProfile(), hourBinDiff);
  nBasicClusterSizeHourly_->Fill(hourDiff,nbcc);

  // --------------------------------------------------
  // SuperClusters
  // --------------------------------------------------
  int nscc = 0;
  float nscc_size = 0.0;
  edm::Handle<reco::SuperClusterCollection> pSuperClusters;
  if ( e.getByLabel(SuperClusterCollection_, pSuperClusters) ) {
    nscc = pSuperClusters->size();
    for(reco::SuperClusterCollection::const_iterator it = pSuperClusters->begin();
	it != pSuperClusters->end(); it++){
      nscc_size += it->clustersSize();
    }
    if(nscc == 0) nscc_size = 0;
    else nscc_size = nscc_size / nscc;
  }
  else edm::LogWarning("EETrendTask") << SuperClusterCollection_ << " is not available";

  ecaldqm::shift2Right(nSuperClusterMinutely_->getTProfile(), minuteBinDiff);
  nSuperClusterMinutely_->Fill(minuteDiff,nscc);

  ecaldqm::shift2Right(nSuperClusterHourly_->getTProfile(), hourBinDiff);
  nSuperClusterHourly_->Fill(hourDiff,nscc);

  ecaldqm::shift2Right(nSuperClusterSizeMinutely_->getTProfile(), minuteBinDiff);
  nSuperClusterSizeMinutely_->Fill(minuteDiff,nscc);

  ecaldqm::shift2Right(nSuperClusterSizeHourly_->getTProfile(), hourBinDiff);
  nSuperClusterSizeHourly_->Fill(hourDiff,nscc);


  // --------------------------------------------------
  // Integrity errors (sum of collections' sizes)
  // --------------------------------------------------
  //  double errorSum = 0.0;

  // --------------------------------------------------
  // EEDetIdCollection0
  // --------------------------------------------------
  int ndic0 = 0;
  edm::Handle<EEDetIdCollection> ids0;
  if ( e.getByLabel(EEDetIdCollection0_, ids0) ) ndic0 = ids0->size();
  else edm::LogWarning("EETrendTask") << EEDetIdCollection0_ << " is not available";


  // --------------------------------------------------
  // EEDetIdCollection1
  // --------------------------------------------------
  int ndic1 = 0;
  edm::Handle<EEDetIdCollection> ids1;
  if ( e.getByLabel(EEDetIdCollection1_, ids1) ) ndic1 = ids1->size();
  else edm::LogWarning("EETrendTask") << EEDetIdCollection1_ << " is not available";


  // --------------------------------------------------
  // EEDetIdCollection2
  // --------------------------------------------------
  int ndic2 = 0;
  edm::Handle<EEDetIdCollection> ids2;
  if ( e.getByLabel(EEDetIdCollection2_, ids2) ) ndic2 = ids2->size();
  else edm::LogWarning("EETrendTask") << EEDetIdCollection2_ << " is not available";


  // --------------------------------------------------
  // EEDetIdCollection3
  // --------------------------------------------------
  int ndic3 = 0;
  edm::Handle<EEDetIdCollection> ids3;
  if ( e.getByLabel(EEDetIdCollection3_, ids3) ) ndic3 = ids3->size();
  else edm::LogWarning("EETrendTask") << EEDetIdCollection3_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection1
  // --------------------------------------------------
  int neic1 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids1;
  if ( e.getByLabel(EcalElectronicsIdCollection1_, eids1) ) neic1 = eids1->size();
  else edm::LogWarning("EETrendTask") << EcalElectronicsIdCollection1_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection2
  // --------------------------------------------------
  int neic2 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids2;
  if ( e.getByLabel(EcalElectronicsIdCollection2_, eids2) ) neic2 = eids2->size();
  else edm::LogWarning("EETrendTask") << EcalElectronicsIdCollection2_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection3
  // --------------------------------------------------
  int neic3 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids3;
  if ( e.getByLabel(EcalElectronicsIdCollection3_, eids3) ) neic3 = eids3->size();
  else edm::LogWarning("EETrendTask") << EcalElectronicsIdCollection3_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection4
  // --------------------------------------------------
  int neic4 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids4;
  if ( e.getByLabel(EcalElectronicsIdCollection4_, eids4) ) neic4 = eids4->size();
  else edm::LogWarning("EETrendTask") << EcalElectronicsIdCollection4_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection5
  // --------------------------------------------------
  int neic5 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids5;
  if ( e.getByLabel(EcalElectronicsIdCollection5_, eids5) ) neic5 = eids5->size();
  else edm::LogWarning("EETrendTask") << EcalElectronicsIdCollection5_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection6
  // --------------------------------------------------
  int neic6 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids6;
  if ( e.getByLabel(EcalElectronicsIdCollection6_, eids6) ) neic6 = eids6->size();
  else edm::LogWarning("EETrendTask") << EcalElectronicsIdCollection6_ << " is not available";


  // --------------------------------------------------
  // Integrity errors (sum of collections' sizes)
  // --------------------------------------------------
  double errorSum = ndic0 + ndic1 + ndic2 + ndic3 +
    neic1 + neic2 + neic3 + neic4 + neic5 + neic6;

  ecaldqm::shift2Right(nIntegrityErrorMinutely_->getTProfile(), minuteBinDiff);
  nIntegrityErrorMinutely_->Fill(minuteDiff,errorSum);

  ecaldqm::shift2Right(nIntegrityErrorHourly_->getTProfile(), hourBinDiff);
  nIntegrityErrorHourly_->Fill(hourDiff,errorSum);


  // --------------------------------------------------
  // FEDRawDataCollection
  // --------------------------------------------------
  float fedSize = 0.;

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
      if((iDcc >= eem1 && iDcc <= eem2) || (iDcc >= eep1 && iDcc <= eep2)) fedSize += sizeInKB;
    }
  }
  else edm::LogWarning("EETrendTask") << FEDRawDataCollection_ << " is not available";

  fedSize /= (eem2 - eem1 + 1 + eep2 - eep1 + 1);

  ecaldqm::shift2Right(nFEDEERawDataMinutely_->getTProfile(), minuteBinDiff);
  nFEDEERawDataMinutely_->Fill(minuteDiff,fedSize);

  ecaldqm::shift2Right(nFEDEERawDataHourly_->getTProfile(), hourBinDiff);
  nFEDEERawDataHourly_->Fill(hourDiff,fedSize);

  // --------------------------------------------------
  // EESRFlagCollection
  // --------------------------------------------------
  int nsfc = 0;
  edm::Handle<EESrFlagCollection> eeSrFlags;
  if ( e.getByLabel(EESRFlagCollection_,eeSrFlags) ) nsfc = eeSrFlags->size();
  else edm::LogWarning("EETrendTask") << EESRFlagCollection_ << " is not available";

  ecaldqm::shift2Right(nEESRFlagMinutely_->getTProfile(), minuteBinDiff);
  nEESRFlagMinutely_->Fill(minuteDiff,nsfc);

  ecaldqm::shift2Right(nEESRFlagHourly_->getTProfile(), hourBinDiff);
  nEESRFlagHourly_->Fill(hourDiff,nsfc);

}


void EETrendTask::updateTime(){

  last_time_ = current_time_;
  current_time_ = time(NULL);

}

