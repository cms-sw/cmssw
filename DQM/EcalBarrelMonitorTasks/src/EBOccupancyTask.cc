/*
 * \file EBOccupancyTask.cc
 *
 * $Date: 2012/04/27 13:46:02 $
 * $Revision: 1.101 $
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

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBOccupancyTask.h"

EBOccupancyTask::EBOccupancyTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

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
    meEBRecHitEnergy_[i] = 0;
    meSpectrum_[i] = 0;
  }

  meEBRecHitSpectrum_ = 0;

  meEBDigiOccupancy_ = 0;
  meEBDigiOccupancyProjEta_ = 0;
  meEBDigiOccupancyProjPhi_ = 0;

  meEBRecHitOccupancy_ = 0;
  meEBRecHitOccupancyProjEta_ = 0;
  meEBRecHitOccupancyProjPhi_ = 0;

  meEBRecHitOccupancyThr_ = 0;
  meEBRecHitOccupancyProjEtaThr_ = 0;
  meEBRecHitOccupancyProjPhiThr_ = 0;

  meEBTrigPrimDigiOccupancy_ = 0;
  meEBTrigPrimDigiOccupancyProjEta_ = 0;
  meEBTrigPrimDigiOccupancyProjPhi_ = 0;

  meEBTrigPrimDigiOccupancyThr_ = 0;
  meEBTrigPrimDigiOccupancyProjEtaThr_ = 0;
  meEBTrigPrimDigiOccupancyProjPhiThr_ = 0;

  meEBTestPulseDigiOccupancy_ = 0;
  meEBLaserDigiOccupancy_ = 0;
  meEBPedestalDigiOccupancy_ = 0;

  recHitEnergyMin_ = 0.300; // GeV
  trigPrimEtMin_ = 4.; // 4 ADCs == 1 GeV

}

EBOccupancyTask::~EBOccupancyTask(){

}

void EBOccupancyTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBOccupancyTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBOccupancyTask/" + subfolder_);
    dqmStore_->rmdir(prefixME_ + "/EBOccupancyTask");
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
    if ( meEBRecHitEnergy_[i] ) meEBRecHitEnergy_[i]->Reset();
    if ( meSpectrum_[i] ) meSpectrum_[i]->Reset();
  }

  if ( meEBRecHitSpectrum_ ) meEBRecHitSpectrum_->Reset();

  if ( meEBDigiOccupancy_ ) meEBDigiOccupancy_->Reset();
  if ( meEBDigiOccupancyProjEta_ ) meEBDigiOccupancyProjEta_->Reset();
  if ( meEBDigiOccupancyProjPhi_ ) meEBDigiOccupancyProjPhi_->Reset();

  if ( meEBRecHitOccupancy_ ) meEBRecHitOccupancy_->Reset();
  if ( meEBRecHitOccupancyProjEta_ ) meEBRecHitOccupancyProjEta_->Reset();
  if ( meEBRecHitOccupancyProjPhi_ ) meEBRecHitOccupancyProjPhi_->Reset();

  if ( meEBRecHitOccupancyThr_ ) meEBRecHitOccupancyThr_->Reset();
  if ( meEBRecHitOccupancyProjEtaThr_ ) meEBRecHitOccupancyProjEtaThr_->Reset();
  if ( meEBRecHitOccupancyProjPhiThr_ ) meEBRecHitOccupancyProjPhiThr_->Reset();

  if ( meEBTrigPrimDigiOccupancy_ ) meEBTrigPrimDigiOccupancy_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjEta_ ) meEBTrigPrimDigiOccupancyProjEta_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjPhi_ ) meEBTrigPrimDigiOccupancyProjPhi_->Reset();

  if ( meEBTrigPrimDigiOccupancyThr_ ) meEBTrigPrimDigiOccupancyThr_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) meEBTrigPrimDigiOccupancyProjEtaThr_->Reset();
  if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) meEBTrigPrimDigiOccupancyProjPhiThr_->Reset();

  if ( meEBTestPulseDigiOccupancy_ ) meEBTestPulseDigiOccupancy_->Reset();
  if ( meEBLaserDigiOccupancy_ ) meEBLaserDigiOccupancy_->Reset();
  if ( meEBPedestalDigiOccupancy_ ) meEBPedestalDigiOccupancy_->Reset();

}

void EBOccupancyTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBOccupancyTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBOccupancyTask/" + subfolder_);

    for (int i = 0; i < 36; i++) {
      name = "EBOT digi occupancy " + Numbers::sEB(i+1);
      meOccupancy_[i] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meOccupancy_[i]->setAxisTitle("ieta", 1);
      meOccupancy_[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(meOccupancy_[i], i+1);

      name = "EBOT MEM digi occupancy " + Numbers::sEB(i+1);
      meOccupancyMem_[i] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0., 5.);
      meOccupancyMem_[i]->setAxisTitle("pseudo-strip", 1);
      meOccupancyMem_[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meOccupancyMem_[i], i+1);

      name = "EBOT rec hit energy " + Numbers::sEB(i+1);
      meEBRecHitEnergy_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., 100, 0., 100., "s");
      meEBRecHitEnergy_[i]->setAxisTitle("ieta", 1);
      meEBRecHitEnergy_[i]->setAxisTitle("iphi", 2);
      meEBRecHitEnergy_[i]->setAxisTitle("energy (GeV)", 3);
      dqmStore_->tag(meEBRecHitEnergy_[i], i+1);

      name = "EBOT energy spectrum " + Numbers::sEB(i+1);
      meSpectrum_[i] = dqmStore_->book1D(name, name, 100, 0., 1.5);
      meSpectrum_[i]->setAxisTitle("energy (GeV)", 1);
      dqmStore_->tag(meSpectrum_[i], i+1);
    }

    name = "EBOT rec hit spectrum";
    meEBRecHitSpectrum_ = dqmStore_->book1D(name, name, 100, 0., 10.);
    meEBRecHitSpectrum_->setAxisTitle("energy (GeV)", 1);

    name = "EBOT digi occupancy";
    meEBDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBDigiOccupancy_->setAxisTitle("jphi", 1);
    meEBDigiOccupancy_->setAxisTitle("jeta", 2);
    name = "EBOT digi occupancy projection eta";
    meEBDigiOccupancyProjEta_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBDigiOccupancyProjEta_->setAxisTitle("jeta", 1);
    meEBDigiOccupancyProjEta_->setAxisTitle("number of digis", 2);
    name = "EBOT digi occupancy projection phi";
    meEBDigiOccupancyProjPhi_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBDigiOccupancyProjPhi_->setAxisTitle("jphi", 1);
    meEBDigiOccupancyProjPhi_->setAxisTitle("number of digis", 2);

    name = "EBOT rec hit occupancy";
    meEBRecHitOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBRecHitOccupancy_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancy_->setAxisTitle("jeta", 2);
    name = "EBOT rec hit occupancy projection eta";
    meEBRecHitOccupancyProjEta_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBRecHitOccupancyProjEta_->setAxisTitle("jeta", 1);
    meEBRecHitOccupancyProjEta_->setAxisTitle("number of hits", 2);
    name = "EBOT rec hit occupancy projection phi";
    meEBRecHitOccupancyProjPhi_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBRecHitOccupancyProjPhi_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyProjPhi_->setAxisTitle("number of hits", 2);

    name = "EBOT rec hit thr occupancy";
    meEBRecHitOccupancyThr_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBRecHitOccupancyThr_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyThr_->setAxisTitle("jeta", 2);
    name = "EBOT rec hit thr occupancy projection eta";
    meEBRecHitOccupancyProjEtaThr_ = dqmStore_->book1DD(name, name, 34, -85., 85.);
    meEBRecHitOccupancyProjEtaThr_->setAxisTitle("jeta", 1);
    meEBRecHitOccupancyProjEtaThr_->setAxisTitle("number of hits", 2);
    name = "EBOT rec hit thr occupancy projection phi";
    meEBRecHitOccupancyProjPhiThr_ = dqmStore_->book1DD(name, name, 72, 0., 360.);
    meEBRecHitOccupancyProjPhiThr_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyProjPhiThr_->setAxisTitle("number of hits", 2);

    name = "EBOT TP digi occupancy";
    meEBTrigPrimDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 72., 34, -17., 17.);
    meEBTrigPrimDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancy_->setAxisTitle("jeta'", 2);
    name = "EBOT TP digi occupancy projection eta";
    meEBTrigPrimDigiOccupancyProjEta_ = dqmStore_->book1DD(name, name, 34, -17., 17.);
    meEBTrigPrimDigiOccupancyProjEta_->setAxisTitle("jeta'", 1);
    meEBTrigPrimDigiOccupancyProjEta_->setAxisTitle("number of TP digis", 2);
    name = "EBOT TP digi occupancy projection phi";
    meEBTrigPrimDigiOccupancyProjPhi_ = dqmStore_->book1DD(name, name, 72, 0., 72.);
    meEBTrigPrimDigiOccupancyProjPhi_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyProjPhi_->setAxisTitle("number of TP digis", 2);

    name = "EBOT TP digi thr occupancy";
    meEBTrigPrimDigiOccupancyThr_ = dqmStore_->book2D(name, name, 72, 0., 72., 34, -17., 17.);
    meEBTrigPrimDigiOccupancyThr_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyThr_->setAxisTitle("jeta'", 2);
    name = "EBOT TP digi thr occupancy projection eta";
    meEBTrigPrimDigiOccupancyProjEtaThr_ = dqmStore_->book1DD(name, name, 34, -17., 17.);
    meEBTrigPrimDigiOccupancyProjEtaThr_->setAxisTitle("jeta'", 1);
    meEBTrigPrimDigiOccupancyProjEtaThr_->setAxisTitle("number of TP digis", 2);
    name = "EBOT TP digi thr occupancy projection phi";
    meEBTrigPrimDigiOccupancyProjPhiThr_ = dqmStore_->book1DD(name, name, 72, 0., 72.);
    meEBTrigPrimDigiOccupancyProjPhiThr_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyProjPhiThr_->setAxisTitle("number of TP digis", 2);

    name = "EBOT test pulse digi occupancy";
    meEBTestPulseDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBTestPulseDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBTestPulseDigiOccupancy_->setAxisTitle("jeta'", 2);

    name = "EBOT laser digi occupancy";
    meEBLaserDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBLaserDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBLaserDigiOccupancy_->setAxisTitle("jeta'", 2);

    name = "EBOT pedestal digi occupancy";
    meEBPedestalDigiOccupancy_ = dqmStore_->book2D(name, name, 72, 0., 360., 34, -85., 85.);
    meEBPedestalDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBPedestalDigiOccupancy_->setAxisTitle("jeta'", 2);

  }

}

void EBOccupancyTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBOccupancyTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBOccupancyTask/" + subfolder_);

    for (int i = 0; i < 36; i++) {
      if ( meOccupancy_[i] ) dqmStore_->removeElement( meOccupancy_[i]->getName() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dqmStore_->removeElement( meOccupancyMem_[i]->getName() );
      meOccupancyMem_[i] = 0;
      if ( meEBRecHitEnergy_[i] ) dqmStore_->removeElement( meEBRecHitEnergy_[i]->getName() );
      meEBRecHitEnergy_[i] = 0;
      if ( meSpectrum_[i] ) dqmStore_->removeElement( meSpectrum_[i]->getName() );
      meSpectrum_[i] = 0;
    }

    if ( meEBRecHitSpectrum_ ) dqmStore_->removeElement( meEBRecHitSpectrum_->getName() );
    meEBRecHitSpectrum_ = 0;

    if ( meEBDigiOccupancy_ ) dqmStore_->removeElement( meEBDigiOccupancy_->getName() );
    meEBDigiOccupancy_ = 0;
    if ( meEBDigiOccupancyProjEta_ ) dqmStore_->removeElement( meEBDigiOccupancyProjEta_->getName() );
    meEBDigiOccupancyProjEta_ = 0;
    if ( meEBDigiOccupancyProjPhi_ ) dqmStore_->removeElement( meEBDigiOccupancyProjPhi_->getName() );
    meEBDigiOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancy_ ) dqmStore_->removeElement( meEBRecHitOccupancy_->getName() );
    meEBRecHitOccupancy_ = 0;
    if ( meEBRecHitOccupancyProjEta_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjEta_->getName() );
    meEBRecHitOccupancyProjEta_ = 0;
    if ( meEBRecHitOccupancyProjPhi_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjPhi_->getName() );
    meEBRecHitOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancyThr_ ) dqmStore_->removeElement( meEBRecHitOccupancyThr_->getName() );
    meEBRecHitOccupancyThr_ = 0;
    if ( meEBRecHitOccupancyProjEtaThr_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjEtaThr_->getName() );
    meEBRecHitOccupancyProjEtaThr_ = 0;
    if ( meEBRecHitOccupancyProjPhiThr_ ) dqmStore_->removeElement( meEBRecHitOccupancyProjPhiThr_->getName() );
    meEBRecHitOccupancyProjPhiThr_ = 0;

    if ( meEBTrigPrimDigiOccupancy_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancy_->getName() );
    meEBTrigPrimDigiOccupancy_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEta_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjEta_->getName() );
    meEBTrigPrimDigiOccupancyProjEta_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhi_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjPhi_->getName() );
    meEBTrigPrimDigiOccupancyProjPhi_ = 0;

    if ( meEBTrigPrimDigiOccupancyThr_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyThr_->getName() );
    meEBTrigPrimDigiOccupancyThr_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjEtaThr_->getName() );
    meEBTrigPrimDigiOccupancyProjEtaThr_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) dqmStore_->removeElement( meEBTrigPrimDigiOccupancyProjPhiThr_->getName() );
    meEBTrigPrimDigiOccupancyProjPhiThr_ = 0;

    if ( meEBTestPulseDigiOccupancy_ ) dqmStore_->removeElement( meEBTestPulseDigiOccupancy_->getName() );
    meEBTestPulseDigiOccupancy_ = 0;

    if ( meEBLaserDigiOccupancy_ ) dqmStore_->removeElement( meEBLaserDigiOccupancy_->getName() );
    meEBLaserDigiOccupancy_ = 0;

    if ( meEBPedestalDigiOccupancy_ ) dqmStore_->removeElement( meEBPedestalDigiOccupancy_->getName() );
    meEBPedestalDigiOccupancy_ = 0;

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

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDetId id = digiItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

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

      if ( runType[ism-1] == testpulse ) {

        if ( meEBTestPulseDigiOccupancy_ ) meEBTestPulseDigiOccupancy_->Fill( xebphi, xebeta );

      }

      if ( runType[ism-1] == laser ) {

        if ( meEBLaserDigiOccupancy_ ) meEBLaserDigiOccupancy_->Fill( xebphi, xebeta );

      }

      if ( runType[ism-1] == pedestal ) {

        if ( meEBPedestalDigiOccupancy_ ) meEBPedestalDigiOccupancy_->Fill( xebphi, xebeta );

      }

    }

  } else {

    edm::LogWarning("EBOccupancyTask") << EBDigiCollection_ << " not available";

  }

  edm::Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      if ( Numbers::subDet( pnItr->id() ) != EcalBarrel ) continue;

      int   ism   = Numbers::iSM( pnItr->id() );

      float PnId  = pnItr->id().iPnId();

      PnId        = PnId - 0.5;
      float st    = 0.0;

      for (int chInStrip = 1; chInStrip <= 5; chInStrip++){
        if ( meOccupancyMem_[ism-1] ) {
          st = chInStrip - 0.5;
          meOccupancyMem_[ism-1]->Fill(PnId, st);
        }
      }

    }

  } else {

    edm::LogWarning("EBOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  c.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

  edm::Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EBDetId id = rechitItr->id();

      // global coordinates
      int ebeta = id.ieta();
      int ebphi = id.iphi();

      float xebeta = ebeta - 0.5*id.zside();
      float xebphi = ebphi - 0.5;

      int ism = Numbers::iSM( id );

      // local coordinates
      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( runType[ism-1] == physics || runType[ism-1] == notdata ) {

        if ( meEBRecHitOccupancy_ ) meEBRecHitOccupancy_->Fill( xebphi, xebeta );
        if ( meEBRecHitOccupancyProjEta_ ) meEBRecHitOccupancyProjEta_->Fill( xebeta );
        if ( meEBRecHitOccupancyProjPhi_ ) meEBRecHitOccupancyProjPhi_->Fill( xebphi );

        uint32_t flag = rechitItr->recoFlag();
	
        uint32_t sev = sevlv->severityLevel( id, *rechits);

        if ( rechitItr->energy() > recHitEnergyMin_ && flag == EcalRecHit::kGood && sev == EcalSeverityLevel::kGood ) {

          if ( meEBRecHitOccupancyThr_ ) meEBRecHitOccupancyThr_->Fill( xebphi, xebeta );
          if ( meEBRecHitOccupancyProjEtaThr_ ) meEBRecHitOccupancyProjEtaThr_->Fill( xebeta );
          if ( meEBRecHitOccupancyProjPhiThr_ ) meEBRecHitOccupancyProjPhiThr_->Fill( xebphi );

        }

        if ( flag == EcalRecHit::kGood && sev == EcalSeverityLevel::kGood ) {
          if ( meEBRecHitEnergy_[ism-1] ) meEBRecHitEnergy_[ism-1]->Fill( xie, xip, rechitItr->energy() );
          if ( meSpectrum_[ism-1] ) meSpectrum_[ism-1]->Fill( rechitItr->energy() );
          if ( meEBRecHitSpectrum_ ) meEBRecHitSpectrum_->Fill( rechitItr->energy() );
        }

      }
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

      float xebeta = ebeta-0.5*tpdigiItr->id().zside();
      float xebphi = ebphi-0.5;

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

