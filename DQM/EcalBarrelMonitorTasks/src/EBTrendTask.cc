/*
 * \file EBTrendTask.cc
 *
 * $Date: 2011/08/30 09:30:33 $
 * $Revision: 1.10 $
 * \author Dongwook Jang, Soon Yung Jun
 *
*/

#include <iostream>
#include <sstream>
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
#include "DQM/EcalCommon/interface/UtilFunctions.h"

#include "TProfile.h"

EBTrendTask::EBTrendTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

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

  ievt_ = 0;

  start_time_ = 0;
  current_time_ = 0;
  last_time_ = 0;
}


EBTrendTask::~EBTrendTask(){
}


void EBTrendTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");
    dqmStore_->rmdir(prefixME_ + "/Trend");
  }

}


void EBTrendTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  start_time_ = time(NULL);

}


void EBTrendTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

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

  std::string name;
  std::string binning;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend");

    // minutely
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend/ShortTerm");

    binning = "5 min bin EB";

    name = "TrendTask num digis " + binning;
    nEBDigiMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEBDigiMinutely_->setAxisTitle("Minutes", 1);
    nEBDigiMinutely_->setAxisTitle("Average Number of EBDigi / 5 minutes", 2);

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
    nFEDEBRawDataMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nFEDEBRawDataMinutely_->setAxisTitle("Minutes", 1);
    nFEDEBRawDataMinutely_->setAxisTitle("Average FED Size in EB / 5 minutes (kB)", 2);

    name = "TrendTask num SR flags " + binning;
    nEBSRFlagMinutely_ = dqmStore_->bookProfile(name, name, 24, 0.0, 120.0, 0.0, 1.0e6, "s");
    nEBSRFlagMinutely_->setAxisTitle("Minutes", 1);
    nEBSRFlagMinutely_->setAxisTitle("Average Number of EBSRFlag / 5 minutes", 2);

    // hourly
    dqmStore_->setCurrentFolder(prefixME_ + "/Trend/LongTerm");

    binning = "20 min bin EB";

    name = "TrendTask num digis " + binning;
    nEBDigiHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEBDigiHourly_->setAxisTitle("Minutes", 1);
    nEBDigiHourly_->setAxisTitle("Average Number of EBDigi / 20 minutes", 2);

    name = "TrendTask num PN digis " + binning;
    nEcalPnDiodeDigiHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEcalPnDiodeDigiHourly_->setAxisTitle("Minutes", 1);
    nEcalPnDiodeDigiHourly_->setAxisTitle("Average Number of EcalPnDiodeDigi / 20 minutes", 2);

    name = "TrendTask num rec hits " + binning;
    nEcalRecHitHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEcalRecHitHourly_->setAxisTitle("Minutes", 1);
    nEcalRecHitHourly_->setAxisTitle("Average Number of EcalRecHit / 20 minutes", 2);

    name = "TrendTask num TP digis " + binning;
    nEcalTrigPrimDigiHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEcalTrigPrimDigiHourly_->setAxisTitle("Minutes", 1);
    nEcalTrigPrimDigiHourly_->setAxisTitle("Average Number of EcalTrigPrimDigi / 20 minutes", 2);

    name = "TrendTask num BCs " + binning;
    nBasicClusterHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nBasicClusterHourly_->setAxisTitle("Minutes", 1);
    nBasicClusterHourly_->setAxisTitle("Average Number of BasicClusters / 20 minutes", 2);

    name = "TrendTask BC size " + binning;
    nBasicClusterSizeHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nBasicClusterSizeHourly_->setAxisTitle("Minutes", 1);
    nBasicClusterSizeHourly_->setAxisTitle("Average Size of BasicClusters / 20 minutes", 2);

    name = "TrendTask num SCs " + binning;
    nSuperClusterHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nSuperClusterHourly_->setAxisTitle("Minutes", 1);
    nSuperClusterHourly_->setAxisTitle("Average Number of SuperClusters / 20 minutes", 2);

    name = "TrendTask SC size " + binning;
    nSuperClusterSizeHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nSuperClusterSizeHourly_->setAxisTitle("Minutes", 1);
    nSuperClusterSizeHourly_->setAxisTitle("Average Size of SuperClusters / 20 minutes", 2);

    name = "TrendTask num integrity errors " + binning;
    nIntegrityErrorHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nIntegrityErrorHourly_->setAxisTitle("Minutes", 1);
    nIntegrityErrorHourly_->setAxisTitle("Average IntegrityErrors / 20 minutes", 2);

    name = "TrendTask DCC event size " + binning;
    nFEDEBRawDataHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nFEDEBRawDataHourly_->setAxisTitle("Minutes", 1);
    nFEDEBRawDataHourly_->setAxisTitle("Average FED Size in EB / 20 minutes (kB)", 2);

    name = "TrendTask num SR flags " + binning;
    nEBSRFlagHourly_ = dqmStore_->bookProfile(name, name, 24, 0.0, 480.0, 0.0, 1.0e6, "s");
    nEBSRFlagHourly_->setAxisTitle("Minutes", 1);
    nEBSRFlagHourly_->setAxisTitle("Average Number of EBSRFlag / 20 minutes", 2);

  }

}


void EBTrendTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if(nEBDigiMinutely_) dqmStore_->removeElement( nEBDigiMinutely_->getFullname());
    nEBDigiMinutely_ = 0;
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
    if(nFEDEBRawDataMinutely_) dqmStore_->removeElement( nFEDEBRawDataMinutely_->getFullname());
    nFEDEBRawDataMinutely_ = 0;
    if(nEBSRFlagMinutely_) dqmStore_->removeElement( nEBSRFlagMinutely_->getFullname());
    nEBSRFlagMinutely_ =0;

    if(nEBDigiHourly_) dqmStore_->removeElement( nEBDigiHourly_->getFullname());
    nEBDigiHourly_ = 0;
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
    if(nFEDEBRawDataHourly_) dqmStore_->removeElement( nFEDEBRawDataHourly_->getFullname());
    nFEDEBRawDataHourly_ = 0;
    if(nEBSRFlagHourly_) dqmStore_->removeElement( nEBSRFlagHourly_->getFullname());
    nEBSRFlagHourly_ = 0;

  }

  init_ = false;

}


void EBTrendTask::endJob(void){

  edm::LogInfo("EBTrendTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}


void EBTrendTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --------------------------------------------------
  // Collect time information
  // --------------------------------------------------

  updateTime();

  long int shortBinDiff = -1;
  long int shortDiff = -1;
  ecaldqm::calcBins(5,60,start_time_,last_time_,current_time_,shortBinDiff,shortDiff);

  long int longBinDiff = -1;
  long int longDiff = -1;
  ecaldqm::calcBins(20,60,start_time_,last_time_,current_time_,longBinDiff,longDiff);


  // --------------------------------------------------
  // EBDigiCollection
  // --------------------------------------------------
  int ndc = 0;
  edm::Handle<EBDigiCollection> digis;
  if ( e.getByLabel(EBDigiCollection_, digis) ) ndc = digis->size();
  else edm::LogWarning("EBTrendTask") << EBDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEBDigiMinutely_->getTProfile(), shortBinDiff);
  nEBDigiMinutely_->Fill(shortDiff,ndc);

  ecaldqm::shift2Right(nEBDigiHourly_->getTProfile(), longBinDiff);
  nEBDigiHourly_->Fill(longDiff,ndc);


  // --------------------------------------------------
  // EcalPnDiodeDigiCollection
  // --------------------------------------------------
  int npdc = 0;
  edm::Handle<EcalPnDiodeDigiCollection> pns;
  if ( e.getByLabel(EcalPnDiodeDigiCollection_, pns) ) npdc = pns->size();
  else edm::LogWarning("EBTrendTask") << EcalPnDiodeDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalPnDiodeDigiMinutely_->getTProfile(), shortBinDiff);
  nEcalPnDiodeDigiMinutely_->Fill(shortDiff,npdc);

  ecaldqm::shift2Right(nEcalPnDiodeDigiHourly_->getTProfile(), longBinDiff);
  nEcalPnDiodeDigiHourly_->Fill(longDiff,npdc);


  // --------------------------------------------------
  // EcalRecHitCollection
  // --------------------------------------------------
  int nrhc = 0;
  edm::Handle<EcalRecHitCollection> hits;
  if ( e.getByLabel(EcalRecHitCollection_, hits) ) nrhc = hits->size();
  else edm::LogWarning("EBTrendTask") << EcalRecHitCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalRecHitMinutely_->getTProfile(), shortBinDiff);
  nEcalRecHitMinutely_->Fill(shortDiff,nrhc);

  ecaldqm::shift2Right(nEcalRecHitHourly_->getTProfile(), longBinDiff);
  nEcalRecHitHourly_->Fill(longDiff,nrhc);

  // --------------------------------------------------
  // EcalTrigPrimDigiCollection
  // --------------------------------------------------
  int ntpdc = 0;
  edm::Handle<EcalTrigPrimDigiCollection> tpdigis;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, tpdigis) ) ntpdc = tpdigis->size();
  else edm::LogWarning("EBTrendTask") << EcalTrigPrimDigiCollection_ << " is not available";

  ecaldqm::shift2Right(nEcalTrigPrimDigiMinutely_->getTProfile(), shortBinDiff);
  nEcalTrigPrimDigiMinutely_->Fill(shortDiff,ntpdc);

  ecaldqm::shift2Right(nEcalTrigPrimDigiHourly_->getTProfile(), longBinDiff);
  nEcalTrigPrimDigiHourly_->Fill(longDiff,ntpdc);

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
  else edm::LogWarning("EBTrendTask") << BasicClusterCollection_ << " is not available";

  ecaldqm::shift2Right(nBasicClusterMinutely_->getTProfile(), shortBinDiff);
  nBasicClusterMinutely_->Fill(shortDiff,nbcc);

  ecaldqm::shift2Right(nBasicClusterHourly_->getTProfile(), longBinDiff);
  nBasicClusterHourly_->Fill(longDiff,nbcc);

  ecaldqm::shift2Right(nBasicClusterSizeMinutely_->getTProfile(), shortBinDiff);
  nBasicClusterSizeMinutely_->Fill(shortDiff,nbcc);

  ecaldqm::shift2Right(nBasicClusterSizeHourly_->getTProfile(), longBinDiff);
  nBasicClusterSizeHourly_->Fill(longDiff,nbcc);

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
  else edm::LogWarning("EBTrendTask") << SuperClusterCollection_ << " is not available";

  ecaldqm::shift2Right(nSuperClusterMinutely_->getTProfile(), shortBinDiff);
  nSuperClusterMinutely_->Fill(shortDiff,nscc);

  ecaldqm::shift2Right(nSuperClusterHourly_->getTProfile(), longBinDiff);
  nSuperClusterHourly_->Fill(longDiff,nscc);

  ecaldqm::shift2Right(nSuperClusterSizeMinutely_->getTProfile(), shortBinDiff);
  nSuperClusterSizeMinutely_->Fill(shortDiff,nscc);

  ecaldqm::shift2Right(nSuperClusterSizeHourly_->getTProfile(), longBinDiff);
  nSuperClusterSizeHourly_->Fill(longDiff,nscc);


  // --------------------------------------------------
  // Integrity errors (sum of collections' sizes)
  // --------------------------------------------------
  //  double errorSum = 0.0;

  // --------------------------------------------------
  // EBDetIdCollection0
  // --------------------------------------------------
  int ndic0 = 0;
  edm::Handle<EBDetIdCollection> ids0;
  if ( e.getByLabel(EBDetIdCollection0_, ids0) ) ndic0 = ids0->size();
  else edm::LogWarning("EBTrendTask") << EBDetIdCollection0_ << " is not available";


  // --------------------------------------------------
  // EBDetIdCollection1
  // --------------------------------------------------
  int ndic1 = 0;
  edm::Handle<EBDetIdCollection> ids1;
  if ( e.getByLabel(EBDetIdCollection1_, ids1) ) ndic1 = ids1->size();
  else edm::LogWarning("EBTrendTask") << EBDetIdCollection1_ << " is not available";


  // --------------------------------------------------
  // EBDetIdCollection2
  // --------------------------------------------------
  int ndic2 = 0;
  edm::Handle<EBDetIdCollection> ids2;
  if ( e.getByLabel(EBDetIdCollection2_, ids2) ) ndic2 = ids2->size();
  else edm::LogWarning("EBTrendTask") << EBDetIdCollection2_ << " is not available";


  // --------------------------------------------------
  // EBDetIdCollection3
  // --------------------------------------------------
  int ndic3 = 0;
  edm::Handle<EBDetIdCollection> ids3;
  if ( e.getByLabel(EBDetIdCollection3_, ids3) ) ndic3 = ids3->size();
  else edm::LogWarning("EBTrendTask") << EBDetIdCollection3_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection1
  // --------------------------------------------------
  int neic1 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids1;
  if ( e.getByLabel(EcalElectronicsIdCollection1_, eids1) ) neic1 = eids1->size();
  else edm::LogWarning("EBTrendTask") << EcalElectronicsIdCollection1_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection2
  // --------------------------------------------------
  int neic2 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids2;
  if ( e.getByLabel(EcalElectronicsIdCollection2_, eids2) ) neic2 = eids2->size();
  else edm::LogWarning("EBTrendTask") << EcalElectronicsIdCollection2_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection3
  // --------------------------------------------------
  int neic3 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids3;
  if ( e.getByLabel(EcalElectronicsIdCollection3_, eids3) ) neic3 = eids3->size();
  else edm::LogWarning("EBTrendTask") << EcalElectronicsIdCollection3_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection4
  // --------------------------------------------------
  int neic4 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids4;
  if ( e.getByLabel(EcalElectronicsIdCollection4_, eids4) ) neic4 = eids4->size();
  else edm::LogWarning("EBTrendTask") << EcalElectronicsIdCollection4_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection5
  // --------------------------------------------------
  int neic5 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids5;
  if ( e.getByLabel(EcalElectronicsIdCollection5_, eids5) ) neic5 = eids5->size();
  else edm::LogWarning("EBTrendTask") << EcalElectronicsIdCollection5_ << " is not available";


  // --------------------------------------------------
  // EcalElectronicsIdCollection6
  // --------------------------------------------------
  int neic6 = 0;
  edm::Handle<EcalElectronicsIdCollection> eids6;
  if ( e.getByLabel(EcalElectronicsIdCollection6_, eids6) ) neic6 = eids6->size();
  else edm::LogWarning("EBTrendTask") << EcalElectronicsIdCollection6_ << " is not available";


  // --------------------------------------------------
  // Integrity errors (sum of collections' sizes)
  // --------------------------------------------------
  double errorSum = ndic0 + ndic1 + ndic2 + ndic3 +
    neic1 + neic2 + neic3 + neic4 + neic5 + neic6;

  ecaldqm::shift2Right(nIntegrityErrorMinutely_->getTProfile(), shortBinDiff);
  nIntegrityErrorMinutely_->Fill(shortDiff,errorSum);

  ecaldqm::shift2Right(nIntegrityErrorHourly_->getTProfile(), longBinDiff);
  nIntegrityErrorHourly_->Fill(longDiff,errorSum);

  // --------------------------------------------------
  // FEDRawDataCollection
  // --------------------------------------------------
  float fedSize      = 0.;

  // Barrel FEDs : 610 - 645
  // Endcap FEDs : 601-609 (EE-) and 646-654 (EE+)
  int eb1 = 610;
  int eb2 = 645;
  int kByte = 1024;

  edm::Handle<FEDRawDataCollection> allFedRawData;
  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {
    for ( int iDcc = eb1; iDcc <= eb2; ++iDcc ) {
      int sizeInKB = allFedRawData->FEDData(iDcc).size()/kByte;
      fedSize += sizeInKB;
    }
  }
  else edm::LogWarning("EBTrendTask") << FEDRawDataCollection_ << " is not available";

  fedSize /= (eb2 - eb1 + 1);

  ecaldqm::shift2Right(nFEDEBRawDataMinutely_->getTProfile(), shortBinDiff);
  nFEDEBRawDataMinutely_->Fill(shortDiff,fedSize);

  ecaldqm::shift2Right(nFEDEBRawDataHourly_->getTProfile(), longBinDiff);
  nFEDEBRawDataHourly_->Fill(longDiff,fedSize);


  // --------------------------------------------------
  // EBSRFlagCollection
  // --------------------------------------------------
  int nsfc = 0;
  edm::Handle<EBSrFlagCollection> ebSrFlags;
  if ( e.getByLabel(EBSRFlagCollection_,ebSrFlags) ) nsfc = ebSrFlags->size();
  else edm::LogWarning("EBTrendTask") << EBSRFlagCollection_ << " is not available";

  ecaldqm::shift2Right(nEBSRFlagMinutely_->getTProfile(), shortBinDiff);
  nEBSRFlagMinutely_->Fill(shortDiff,nsfc);

  ecaldqm::shift2Right(nEBSRFlagHourly_->getTProfile(), longBinDiff);
  nEBSRFlagHourly_->Fill(longDiff,nsfc);


  if(verbose_){
    printf("run(%d), event(%d), ndc(%d), npdc(%d), nrhc(%d), ntpdc(%d), nbcc(%d), ",
	   e.id().run(),e.id().event(), ndc, npdc, nrhc, ntpdc, nbcc);
    printf("nscc(%d), ndic0(%d), ndic1(%d), ndic2(%d), ndic3(%d), neic1(%d), neic2(%d), neic3(%d), ",
	   nscc, ndic0, ndic1, ndic2, ndic3, neic1, neic2, neic3);
    printf("neic4(%d), neic5(%d), neic6(%d), errorSum(%f), nsfc(%d), ",
	   neic4, neic5, neic6, errorSum, nsfc);
  }

}


void EBTrendTask::updateTime(){

  last_time_ = current_time_;
  current_time_ = time(NULL);

}

