/*
 * \file EBOccupancyTask.cc
 *
 * $Date: 2011/10/30 15:46:27 $
 * $Revision: 1.98 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBOccupancyTask.h"

EBOccupancyTask::EBOccupancyTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  for (int i = 0; i < 36; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
    meEvent_[i] = 0;
  }

  meDCCOccupancy_ = 0;

  meEBDigiOccupancy_ = 0;
  meEBDigiOccupancyProjEta_ = 0;
  meEBDigiOccupancyProjPhi_ = 0;

  meEBDigiNumber_ = 0;
  meEBDigiNumberPerFED_ = 0;

  meEBRecHitOccupancy_ = 0;
  meEBRecHitOccupancyProjEta_ = 0;
  meEBRecHitOccupancyProjPhi_ = 0;

  meEBRecHitOccupancyThr_ = 0;
  meEBRecHitOccupancyProjEtaThr_ = 0;
  meEBRecHitOccupancyProjPhiThr_ = 0;

  meEBRecHitNumber_ = 0;
  meEBRecHitNumberPerFED_ = 0;

  meEBTrigPrimDigiOccupancy_ = 0;
  meEBTrigPrimDigiOccupancyProjEta_ = 0;
  meEBTrigPrimDigiOccupancyProjPhi_ = 0;

  meEBTrigPrimDigiOccupancyThr_ = 0;
  meEBTrigPrimDigiOccupancyProjEtaThr_ = 0;
  meEBTrigPrimDigiOccupancyProjPhiThr_ = 0;

  recHitEnergyMin_ = 0.300; // GeV
  trigPrimEtMin_ = 4.; // 2 ADCs == 1 GeV

  ievt_ = 0;

}

EBOccupancyTask::~EBOccupancyTask(){

}

void EBOccupancyTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Occupancy");
    dqmStore_->rmdir(prefixME_ + "/Occupancy");
  }

}

void EBOccupancyTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBOccupancyTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBOccupancyTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if ( meOccupancy_[i] ) meOccupancy_[i]->Reset();
    if ( meOccupancyMem_[i] ) meOccupancyMem_[i]->Reset();
  }

  if( meDCCOccupancy_ ) meDCCOccupancy_->Reset();

  if ( meEBDigiOccupancy_ ) meEBDigiOccupancy_->Reset();
  if ( meEBDigiOccupancyProjEta_ ) meEBDigiOccupancyProjEta_->Reset();
  if ( meEBDigiOccupancyProjPhi_ ) meEBDigiOccupancyProjPhi_->Reset();

  if(meEBDigiNumber_) meEBDigiNumber_->Reset();
  if(meEBDigiNumberPerFED_) meEBDigiNumberPerFED_->Reset();

  if ( meEBRecHitOccupancy_ ) meEBRecHitOccupancy_->Reset();
  if ( meEBRecHitOccupancyProjEta_ ) meEBRecHitOccupancyProjEta_->Reset();
  if ( meEBRecHitOccupancyProjPhi_ ) meEBRecHitOccupancyProjPhi_->Reset();

  if(meEBRecHitNumber_) meEBRecHitNumber_->Reset();
  if(meEBRecHitNumberPerFED_) meEBRecHitNumberPerFED_->Reset();

  if ( meEBRecHitOccupancyThr_ ) meEBRecHitOccupancyThr_->Reset();
  if ( meEBRecHitOccupancyProjEtaThr_ ) meEBRecHitOccupancyProjEtaThr_->Reset();
  if ( meEBRecHitOccupancyProjPhiThr_ ) meEBRecHitOccupancyProjPhiThr_->Reset();

  if ( meEBTrigPrimDigiOccupancy_ ) meEBTrigPrimDigiOccupancy_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjEta_ ) meEBTrigPrimDigiOccupancyProjEta_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjPhi_ ) meEBTrigPrimDigiOccupancyProjPhi_->Reset();

  if ( meEBTrigPrimDigiOccupancyThr_ ) meEBTrigPrimDigiOccupancyThr_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) meEBTrigPrimDigiOccupancyProjEtaThr_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) meEBTrigPrimDigiOccupancyProjPhiThr_->Reset();

}

void EBOccupancyTask::setup(void){

  init_ = true;

  std::string name;
  std::string dir;

  if ( dqmStore_ ) {
    dir = prefixME_ + "/Occupancy";
    dqmStore_->setCurrentFolder(dir);

    name = "OccupancyTask DCC occupancy EB";
    meDCCOccupancy_ = dqmStore_->book1D(name, name, 36, 0., 36.);
    for (int i = 0; i < 36; i++) {
      meDCCOccupancy_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    dqmStore_->setCurrentFolder(dir + "/Digi");
    for (int i = 0; i < 36; i++) {
      name = "OccupancyTask digi occupancy " + Numbers::sEB(i+1);
      meOccupancy_[i] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meOccupancy_[i]->setAxisTitle("ieta", 1);
      meOccupancy_[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(meOccupancy_[i], i+1);
    }

    name = "OccupancyTask digi occupancy EB";
    meEBDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBDigiOccupancy_->setAxisTitle("jphi", 1);
    meEBDigiOccupancy_->setAxisTitle("jeta", 2);
    name = "OccupancyTask digi occupancy eta EB";
    meEBDigiOccupancyProjEta_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBDigiOccupancyProjEta_->setAxisTitle("jeta", 1);
    meEBDigiOccupancyProjEta_->setAxisTitle("number of digis", 2);
    name = "OccupancyTask digi occupancy phi EB";
    meEBDigiOccupancyProjPhi_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBDigiOccupancyProjPhi_->setAxisTitle("jphi", 1);
    meEBDigiOccupancyProjPhi_->setAxisTitle("number of digis", 2);

    name = "OccupancyTask digi number EB";
    meEBDigiNumber_ = dqmStore_->book1D(name, name, 100, 0., 5000.);

    name = "OccupancyTask digi number profile EB";
    meEBDigiNumberPerFED_ = dqmStore_->bookProfile(name, name, 36, 0., 36., 0., 1700.);
    for (int i = 0; i < 36; i++) {
      meEBDigiNumberPerFED_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    dqmStore_->setCurrentFolder(dir + "/MEMDigi");
    for (int i = 0; i < 36; i++) {
      name = "OccupancyTask MEM digi occupancy " + Numbers::sEB(i+1);
      meOccupancyMem_[i] = dqmStore_->book2D(name, name, 10, 0., 10., 1, 0., 5.);
      meOccupancyMem_[i]->setAxisTitle("channel", 1);
      dqmStore_->tag(meOccupancyMem_[i], i+1);

    }

    dqmStore_->setCurrentFolder(dir + "/RecHit");
    name = "OccupancyTask rec hit occupancy EB";
    meEBRecHitOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBRecHitOccupancy_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancy_->setAxisTitle("jeta", 2);
    name = "OccupancyTask rec hit occupancy eta EB";
    meEBRecHitOccupancyProjEta_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBRecHitOccupancyProjEta_->setAxisTitle("jeta", 1);
    meEBRecHitOccupancyProjEta_->setAxisTitle("number of hits", 2);
    name = "OccupancyTask rec hit occupancy phi EB";
    meEBRecHitOccupancyProjPhi_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBRecHitOccupancyProjPhi_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyProjPhi_->setAxisTitle("number of hits", 2);

    name = "OccupancyTask rec hit number EB";
    meEBRecHitNumber_ = dqmStore_->book1D(name, name, 100, 0., 5000.);

    name = "OccupancyTask rec hit number profile EB";
    meEBRecHitNumberPerFED_ = dqmStore_->bookProfile(name, name, 36, 0., 36., 0., 1700.);
    for (int i = 0; i < 36; i++) {
      meEBRecHitNumberPerFED_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    dqmStore_->setCurrentFolder(dir + "/RecHitThres");
    name = "OccupancyTask rec hit thres occupancy EB";
    meEBRecHitOccupancyThr_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBRecHitOccupancyThr_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyThr_->setAxisTitle("jeta", 2);
    name = "OccupancyTask rec hit thres occupancy eta EB";
    meEBRecHitOccupancyProjEtaThr_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBRecHitOccupancyProjEtaThr_->setAxisTitle("jeta", 1);
    meEBRecHitOccupancyProjEtaThr_->setAxisTitle("number of hits", 2);
    name = "OccupancyTask rec hit thres occupancy phi EB";
    meEBRecHitOccupancyProjPhiThr_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBRecHitOccupancyProjPhiThr_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyProjPhiThr_->setAxisTitle("number of hits", 2);

    dqmStore_->setCurrentFolder(dir + "/TPDigi");
    name = "OccupancyTask TP digi occupancy EB";
    meEBTrigPrimDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBTrigPrimDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancy_->setAxisTitle("jeta'", 2);
    name = "OccupancyTask TP digi occupancy eta EB";
    meEBTrigPrimDigiOccupancyProjEta_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBTrigPrimDigiOccupancyProjEta_->setAxisTitle("jeta'", 1);
    meEBTrigPrimDigiOccupancyProjEta_->setAxisTitle("number of TP digis", 2);
    name = "OccupancyTask TP digi occupancy phi EB";
    meEBTrigPrimDigiOccupancyProjPhi_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBTrigPrimDigiOccupancyProjPhi_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyProjPhi_->setAxisTitle("number of TP digis", 2);

    dqmStore_->setCurrentFolder(dir + "/TPDigiThres");
    name = "OccupancyTask TP digi thres occupancy EB";
    meEBTrigPrimDigiOccupancyThr_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBTrigPrimDigiOccupancyThr_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyThr_->setAxisTitle("jeta'", 2);
    name = "OccupancyTask TP digi thres occupancy eta EB";
    meEBTrigPrimDigiOccupancyProjEtaThr_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBTrigPrimDigiOccupancyProjEtaThr_->setAxisTitle("jeta'", 1);
    meEBTrigPrimDigiOccupancyProjEtaThr_->setAxisTitle("number of TP digis", 2);
    name = "OccupancyTask TP digi thres occupancy phi EB";
    meEBTrigPrimDigiOccupancyProjPhiThr_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBTrigPrimDigiOccupancyProjPhiThr_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyProjPhiThr_->setAxisTitle("number of TP digis", 2);

  }

}

void EBOccupancyTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    for (int i = 0; i < 36; i++) {
      if ( meOccupancy_[i] ) dqmStore_->removeElement( meOccupancy_[i]->getFullname() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dqmStore_->removeElement( meOccupancyMem_[i]->getFullname() );
      meOccupancyMem_[i] = 0;
    }

    if ( meEBDigiOccupancy_ ) dqmStore_->removeElement( meEBDigiOccupancy_->getFullname() );
    meEBDigiOccupancy_ = 0;
    if ( meEBDigiOccupancyProjEta_ ) dqmStore_->removeElement( meEBDigiOccupancyProjEta_->getFullname() );
    meEBDigiOccupancyProjEta_ = 0;
    if ( meEBDigiOccupancyProjPhi_ ) dqmStore_->removeElement( meEBDigiOccupancyProjPhi_->getFullname() );
    meEBDigiOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancy_ ) dqmStore_->removeElement( meEBRecHitOccupancy_->getFullname() );
    meEBRecHitOccupancy_ = 0;
    if ( meEBRecHitOccupancyProjEta_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjEta_->getFullname() );
    meEBRecHitOccupancyProjEta_ = 0;
    if ( meEBRecHitOccupancyProjPhi_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjPhi_->getFullname() );
    meEBRecHitOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancyThr_ ) dqmStore_->removeElement( meEBRecHitOccupancyThr_->getFullname() );
    meEBRecHitOccupancyThr_ = 0;
    if ( meEBRecHitOccupancyProjEtaThr_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjEtaThr_->getFullname() );
    meEBRecHitOccupancyProjEtaThr_ = 0;
    if ( meEBRecHitOccupancyProjPhiThr_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjPhiThr_->getFullname() );
    meEBRecHitOccupancyProjPhiThr_ = 0;

    if ( meEBTrigPrimDigiOccupancy_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancy_->getFullname() );
    meEBTrigPrimDigiOccupancy_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEta_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjEta_->getFullname() );
    meEBTrigPrimDigiOccupancyProjEta_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhi_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjPhi_->getFullname() );
    meEBTrigPrimDigiOccupancyProjPhi_ = 0;

    if ( meEBTrigPrimDigiOccupancyThr_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyThr_->getFullname() );
    meEBTrigPrimDigiOccupancyThr_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjEtaThr_->getFullname() );
    meEBTrigPrimDigiOccupancyProjEtaThr_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjPhiThr_->getFullname() );
    meEBTrigPrimDigiOccupancyProjPhiThr_ = 0;

  }

  init_ = false;

}

void EBOccupancyTask::endJob(void) {

  edm::LogInfo("EBOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBOccupancyTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  int runType[36] = { notdata };

  edm::Handle<EcalRawDataCollection> dcchs;

  if (  e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );

      if(meDCCOccupancy_) meDCCOccupancy_->Fill(ism - 0.5);

      int runtype = dcchItr->getRunType();

      if ( runtype == EcalDCCHeaderBlock::COSMIC ||
           runtype == EcalDCCHeaderBlock::MTCC ||
           runtype == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           runtype == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           runtype == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           runtype == EcalDCCHeaderBlock::PHYSICS_LOCAL ) runType[ism-1] = physics;
      if ( runtype == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
           runtype == EcalDCCHeaderBlock::TESTPULSE_GAP ) runType[ism-1] = testpulse;
      if ( runtype == EcalDCCHeaderBlock::LASER_STD ||
           runtype == EcalDCCHeaderBlock::LASER_GAP ) runType[ism-1] = laser;
      if ( runtype == EcalDCCHeaderBlock::PEDESTAL_STD ||
           runtype == EcalDCCHeaderBlock::PEDESTAL_GAP ) runType[ism-1] = pedestal;

    }

  } else {
    edm::LogWarning("EBOccupancyTask") << EcalRawDataCollection_ << " not available";
  }

  edm::Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " digi collection size " << nebd;

    if(meEBDigiNumber_) meEBDigiNumber_->Fill(nebd);

    std::vector<int> nDigis(36, 0);

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDetId id = digiItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      if(ism >= 1 && ism <= 36) nDigis[ism - 1] += 1;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
        edm::LogWarning("EBOccupancyTask") << " det id = " << id;
        edm::LogWarning("EBOccupancyTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;
        edm::LogWarning("EBOccupancyTask") << " xie, xip " << xie << " " << xip;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill(xie, xip);

      int ebeta = id.ieta();
      int ebphi = id.iphi();

      float xebeta = ebeta - 0.5*id.zside();
      float xebphi = ebphi - 0.5;

      if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

        if ( meEBDigiOccupancy_ ) meEBDigiOccupancy_->Fill( xebphi, xebeta );
        if ( meEBDigiOccupancyProjEta_ ) meEBDigiOccupancyProjEta_->Fill( xebeta );
        if ( meEBDigiOccupancyProjPhi_ ) meEBDigiOccupancyProjPhi_->Fill( xebphi );

      }

    }

    if(meEBDigiNumberPerFED_){
      for(int iSM(0); iSM < 36; iSM++)
	meEBDigiNumberPerFED_->Fill(iSM - 0.5, nDigis[iSM]);
    }
  } else {

    edm::LogWarning("EBOccupancyTask") << EBDigiCollection_ << " not available";

  }

  edm::Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalBarrel ) continue;

      int   ism   = Numbers::iSM( pnItr->id() );

      float PnId  = pnItr->id().iPnId();

      PnId        = PnId - 0.5;

      if ( meOccupancyMem_[ism-1] ) meOccupancyMem_[ism-1]->Fill(PnId, 0.5);

    }

  } else {

    edm::LogWarning("EBOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  edm::Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    if(meEBRecHitNumber_) meEBRecHitNumber_->Fill(nebrh);

    std::vector<int> nHits(36, 0);

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EBDetId id = rechitItr->id();

      // global coordinates
      int ebeta = id.ieta();
      int ebphi = id.iphi();

      float xebeta = ebeta - 0.5*id.zside();
      float xebphi = ebphi - 0.5;

      int ism = Numbers::iSM( id );

      if(ism >= 1 && ism <= 36) nHits[ism - 1] += 1;

      if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

        if ( meEBRecHitOccupancy_ ) meEBRecHitOccupancy_->Fill( xebphi, xebeta );
        if ( meEBRecHitOccupancyProjEta_ ) meEBRecHitOccupancyProjEta_->Fill( xebeta );
        if ( meEBRecHitOccupancyProjPhi_ ) meEBRecHitOccupancyProjPhi_->Fill( xebphi );

	// it's no use to use severitylevel to detect spikes (SeverityLevelAlgo simply uses RecHit flag for spikes)
	uint32_t mask = 0xffffffff ^ ((0x1 << EcalRecHit::kGood));

	if( !rechitItr->checkFlagMask(mask) ){
	  if ( rechitItr->energy() > recHitEnergyMin_ ){
	    if ( meEBRecHitOccupancyThr_ ) meEBRecHitOccupancyThr_->Fill( xebphi, xebeta );
	    if ( meEBRecHitOccupancyProjEtaThr_ ) meEBRecHitOccupancyProjEtaThr_->Fill( xebeta );
	    if ( meEBRecHitOccupancyProjPhiThr_ ) meEBRecHitOccupancyProjPhiThr_->Fill( xebphi );
	  }

        }

      }
    }

    if(meEBRecHitNumberPerFED_){
      for(int iSM(0); iSM < 36; iSM++)
	meEBRecHitNumberPerFED_->Fill(iSM - 0.5, nHits[iSM]);
    }

  } else {

    edm::LogWarning("EBOccupancyTask") << EcalRecHitCollection_ << " not available";

  }

  edm::Handle<EcalTrigPrimDigiCollection> trigPrimDigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, trigPrimDigis) ) {

    int nebtpg = trigPrimDigis->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " trigger primitives digis collection size " << nebtpg;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = trigPrimDigis->begin(); tpdigiItr != trigPrimDigis->end(); ++tpdigiItr ) {

      if ( Numbers::subDet( tpdigiItr->id() ) != EcalBarrel ) continue;

      int ebeta = tpdigiItr->id().ieta();
      int ebphi = tpdigiItr->id().iphi();

      // phi_tower: change the range from global to SM-local
      // phi==0 is in the middle of a SM
      ebphi = ebphi + 2;
      if ( ebphi > 72 ) ebphi = ebphi - 72;

      float xebeta = (ebeta-0.5*tpdigiItr->id().zside())*5.;
      float xebphi = (ebphi-0.5)*5;

      int ism = Numbers::iSM( tpdigiItr->id() );

      if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

        if ( meEBTrigPrimDigiOccupancy_ ) meEBTrigPrimDigiOccupancy_->Fill( xebphi, xebeta );
        if ( meEBTrigPrimDigiOccupancyProjEta_ ) meEBTrigPrimDigiOccupancyProjEta_->Fill( xebeta );
        if ( meEBTrigPrimDigiOccupancyProjPhi_ ) meEBTrigPrimDigiOccupancyProjPhi_->Fill( xebphi );

        if ( tpdigiItr->compressedEt() > trigPrimEtMin_ ) {

          if ( meEBTrigPrimDigiOccupancyThr_ ) meEBTrigPrimDigiOccupancyThr_->Fill( xebphi, xebeta );
          if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) meEBTrigPrimDigiOccupancyProjEtaThr_->Fill( xebeta );
          if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) meEBTrigPrimDigiOccupancyProjPhiThr_->Fill( xebphi );

        }
      }
    }

  } else {

    edm::LogWarning("EBOccupancyTask") << EcalTrigPrimDigiCollection_ << " not available";

  }

}

