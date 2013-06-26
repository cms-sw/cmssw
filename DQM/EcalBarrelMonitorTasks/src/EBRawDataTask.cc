/*
 * \file EBRawDataTask.cc
 *
 * $Date: 2012/06/28 12:14:29 $
 * $Revision: 1.48 $
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/FEDRawData/src/fed_header.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBRawDataTask.h"

EBRawDataTask::EBRawDataTask(const edm::ParameterSet& ps) {

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  meEBEventTypePreCalibrationBX_ = 0;
  meEBEventTypeCalibrationBX_ = 0;
  meEBEventTypePostCalibrationBX_ = 0;
  meEBCRCErrors_ = 0;
  meEBRunNumberErrors_ = 0;
  meEBOrbitNumberErrors_ = 0;
  meEBTriggerTypeErrors_ = 0;
  meEBCalibrationEventErrors_ = 0;
  meEBL1ADCCErrors_ = 0;
  meEBBunchCrossingDCCErrors_ = 0;
  meEBL1AFEErrors_ = 0;
  meEBBunchCrossingFEErrors_ = 0;
  meEBL1ATCCErrors_ = 0;
  meEBBunchCrossingTCCErrors_ = 0;
  meEBL1ASRPErrors_ = 0;
  meEBBunchCrossingSRPErrors_ = 0;

  meEBSynchronizationErrorsByLumi_ = 0;

  meEBSynchronizationErrorsTrend_ = 0;

  calibrationBX_ = 3490;

}

EBRawDataTask::~EBRawDataTask() {
}

void EBRawDataTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask/" + subfolder_);
    dqmStore_->rmdir(prefixME_ + "/EBRawDataTask");
  }

}

void EBRawDataTask::beginLuminosityBlock(const edm::LuminosityBlock& _lumi, const  edm::EventSetup&) {

  if ( ! init_ ) this->setup();

  ls_ = _lumi.luminosityBlock();

  if ( meEBSynchronizationErrorsByLumi_ ) meEBSynchronizationErrorsByLumi_->Reset();

  if ( meEBSynchronizationErrorsTrend_ ){
    int bin(meEBSynchronizationErrorsTrend_->getTH1()->GetXaxis()->FindBin(ls_ - 0.5));
    meEBSynchronizationErrorsTrend_->setBinContent(bin - 1, fatalErrors_);
    meEBSynchronizationErrorsTrend_->getTH1()->SetEntries(fatalErrors_);
  }

}

void EBRawDataTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  fatalErrors_ = 0.;

  if ( meEBSynchronizationErrorsTrend_ ){
    meEBSynchronizationErrorsTrend_->getTH1()->GetXaxis()->SetLimits(0., 50.);
  }
}

void EBRawDataTask::endRun(const edm::Run& r, const edm::EventSetup& c) {
}

void EBRawDataTask::reset(void) {

  if ( meEBEventTypePreCalibrationBX_ ) meEBEventTypePreCalibrationBX_->Reset();
  if ( meEBEventTypeCalibrationBX_ ) meEBEventTypeCalibrationBX_->Reset();
  if ( meEBEventTypePostCalibrationBX_ ) meEBEventTypePostCalibrationBX_->Reset();
  if ( meEBCRCErrors_ ) meEBCRCErrors_->Reset();
  if ( meEBRunNumberErrors_ ) meEBRunNumberErrors_->Reset();
  if ( meEBOrbitNumberErrors_ ) meEBOrbitNumberErrors_->Reset();
  if ( meEBTriggerTypeErrors_ ) meEBTriggerTypeErrors_->Reset();
  if ( meEBCalibrationEventErrors_ ) meEBCalibrationEventErrors_->Reset();
  if ( meEBL1ADCCErrors_ ) meEBL1ADCCErrors_->Reset();
  if ( meEBBunchCrossingDCCErrors_ ) meEBBunchCrossingDCCErrors_->Reset();
  if ( meEBL1AFEErrors_ ) meEBL1AFEErrors_->Reset();
  if ( meEBBunchCrossingFEErrors_ ) meEBBunchCrossingFEErrors_->Reset();
  if ( meEBL1ATCCErrors_ ) meEBL1ATCCErrors_->Reset();
  if ( meEBBunchCrossingTCCErrors_ ) meEBBunchCrossingTCCErrors_->Reset();
  if ( meEBL1ASRPErrors_ ) meEBL1ASRPErrors_->Reset();
  if ( meEBBunchCrossingSRPErrors_ ) meEBBunchCrossingSRPErrors_->Reset();
  if ( meEBSynchronizationErrorsByLumi_ ) meEBSynchronizationErrorsByLumi_->Reset();
  if ( meEBSynchronizationErrorsTrend_ ) meEBSynchronizationErrorsTrend_->Reset();
}

void EBRawDataTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask/" + subfolder_);

    name = "EBRDT event type pre calibration BX";
    meEBEventTypePreCalibrationBX_ = dqmStore_->book1D(name, name, 31, -1., 30.);
    meEBEventTypePreCalibrationBX_->setBinLabel(1, "UNKNOWN", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEBEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);

    name = "EBRDT event type calibration BX";
    meEBEventTypeCalibrationBX_ = dqmStore_->book1D(name, name, 31, -1., 30.);
    meEBEventTypeCalibrationBX_->setBinLabel(1, "UNKNOWN", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEBEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);

    name = "EBRDT event type post calibration BX";
    meEBEventTypePostCalibrationBX_ = dqmStore_->book1D(name, name, 31, -1., 30.);
    meEBEventTypePostCalibrationBX_->setBinLabel(1, "UNKNOWN", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEBEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);

    name = "EBRDT CRC errors";
    meEBCRCErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBCRCErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT run number errors";
    meEBRunNumberErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBRunNumberErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT orbit number errors";
    meEBOrbitNumberErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBOrbitNumberErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT trigger type errors";
    meEBTriggerTypeErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBTriggerTypeErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT calibration event errors";
    meEBCalibrationEventErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBCalibrationEventErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT L1A DCC errors";
    meEBL1ADCCErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBL1ADCCErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT bunch crossing DCC errors";
    meEBBunchCrossingDCCErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBBunchCrossingDCCErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT L1A FE errors";
    meEBL1AFEErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBL1AFEErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT bunch crossing FE errors";
    meEBBunchCrossingFEErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBBunchCrossingFEErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT L1A TCC errors";
    meEBL1ATCCErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBL1ATCCErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT bunch crossing TCC errors";
    meEBBunchCrossingTCCErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBBunchCrossingTCCErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT L1A SRP errors";
    meEBL1ASRPErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBL1ASRPErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT bunch crossing SRP errors";
    meEBBunchCrossingSRPErrors_ = dqmStore_->book1D(name, name, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBBunchCrossingSRPErrors_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT FE synchronization errors by lumi";
    meEBSynchronizationErrorsByLumi_ = dqmStore_->book1D(name, name, 36, 1, 37);
    meEBSynchronizationErrorsByLumi_->setLumiFlag();
    for (int i = 0; i < 36; i++) {
      meEBSynchronizationErrorsByLumi_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBRDT accumulated FE synchronization errors";
    meEBSynchronizationErrorsTrend_ = dqmStore_->book1D(name, name, 50, 0., 50.);
    meEBSynchronizationErrorsTrend_->setAxisTitle("LumiSection", 1);

  }

}

void EBRawDataTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask/" + subfolder_);

    if ( meEBEventTypePreCalibrationBX_ ) dqmStore_->removeElement( meEBEventTypePreCalibrationBX_->getName() );
    meEBEventTypePreCalibrationBX_ = 0;

    if ( meEBEventTypeCalibrationBX_ ) dqmStore_->removeElement( meEBEventTypeCalibrationBX_->getName() );
    meEBEventTypeCalibrationBX_ = 0;

    if ( meEBEventTypePostCalibrationBX_ ) dqmStore_->removeElement( meEBEventTypePostCalibrationBX_->getName() );
    meEBEventTypePostCalibrationBX_ = 0;

    if ( meEBCRCErrors_ ) dqmStore_->removeElement( meEBCRCErrors_->getName() );
    meEBCRCErrors_ = 0;

    if ( meEBRunNumberErrors_ ) dqmStore_->removeElement( meEBRunNumberErrors_->getName() );
    meEBRunNumberErrors_ = 0;

    if ( meEBOrbitNumberErrors_ ) dqmStore_->removeElement( meEBOrbitNumberErrors_->getName() );
    meEBOrbitNumberErrors_ = 0;

    if ( meEBTriggerTypeErrors_ ) dqmStore_->removeElement( meEBTriggerTypeErrors_->getName() );
    meEBTriggerTypeErrors_ = 0;

    if ( meEBCalibrationEventErrors_ ) dqmStore_->removeElement( meEBCalibrationEventErrors_->getName() );
    meEBCalibrationEventErrors_ = 0;

    if ( meEBL1ADCCErrors_ ) dqmStore_->removeElement( meEBL1ADCCErrors_->getName() );
    meEBL1ADCCErrors_ = 0;

    if ( meEBBunchCrossingDCCErrors_ ) dqmStore_->removeElement( meEBBunchCrossingDCCErrors_->getName() );
    meEBBunchCrossingDCCErrors_ = 0;

    if ( meEBL1AFEErrors_ ) dqmStore_->removeElement( meEBL1AFEErrors_->getName() );
    meEBL1AFEErrors_ = 0;

    if ( meEBBunchCrossingFEErrors_ ) dqmStore_->removeElement( meEBBunchCrossingFEErrors_->getName() );
    meEBBunchCrossingFEErrors_ = 0;

    if ( meEBL1ATCCErrors_ ) dqmStore_->removeElement( meEBL1ATCCErrors_->getName() );
    meEBL1ATCCErrors_ = 0;

    if ( meEBBunchCrossingTCCErrors_ ) dqmStore_->removeElement( meEBBunchCrossingTCCErrors_->getName() );
    meEBBunchCrossingTCCErrors_ = 0;

    if ( meEBL1ASRPErrors_ ) dqmStore_->removeElement( meEBL1ASRPErrors_->getName() );
    meEBL1ASRPErrors_ = 0;

    if ( meEBBunchCrossingSRPErrors_ ) dqmStore_->removeElement( meEBBunchCrossingSRPErrors_->getName() );
    meEBBunchCrossingSRPErrors_ = 0;

    if ( meEBSynchronizationErrorsByLumi_ ) dqmStore_->removeElement( meEBSynchronizationErrorsByLumi_->getName() );
    meEBSynchronizationErrorsByLumi_ = 0;

    if ( meEBSynchronizationErrorsTrend_ ) dqmStore_->removeElement( meEBSynchronizationErrorsTrend_->getName() );
    meEBSynchronizationErrorsTrend_ = 0;
  }

  init_ = false;

}

void EBRawDataTask::endLuminosityBlock(const edm::LuminosityBlock&, const  edm::EventSetup&) {

  MonitorElement* me(meEBSynchronizationErrorsTrend_);
  if(!me) return;
  if(ls_ >= 50){
    for(int ix(1); ix <= 50; ix++)
      me->setBinContent(ix, me->getBinContent(ix + 1));

    me->getTH1()->GetXaxis()->SetLimits(ls_ - 49, ls_ + 1);
  }
}

void EBRawDataTask::endJob(void) {

  edm::LogInfo("EBRawDataTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBRawDataTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  // fill bin 0 with number of events in the lumi
  if ( meEBSynchronizationErrorsByLumi_ ) meEBSynchronizationErrorsByLumi_->Fill(0.);

  float errorsInEvent(0.);

  int evt_runNumber = e.id().run();

  int GT_L1A=0, GT_OrbitNumber=0, GT_BunchCrossing=0, GT_TriggerType=0;

  edm::Handle<FEDRawDataCollection> allFedRawData;

  int gtFedDataSize = 0;

  int ECALDCC_L1A_MostFreqId = -1;
  int ECALDCC_OrbitNumber_MostFreqId = -1;
  int ECALDCC_BunchCrossing_MostFreqId = -1;
  int ECALDCC_TriggerType_MostFreqId = -1;

  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {

    // GT FED data
    const FEDRawData& gtFedData = allFedRawData->FEDData(812);

    gtFedDataSize = gtFedData.size()/sizeof(uint64_t);

    if ( gtFedDataSize > 0 ) {

      FEDHeader header(gtFedData.data());

#define  H_L1_MASK           0xFFFFFF
#define  H_ORBITCOUNTER_MASK 0xFFFFFFFF
#define  H_BX_MASK           0xFFF
#define  H_TTYPE_MASK        0xF

      GT_L1A           = header.lvl1ID()    & H_L1_MASK;
      GT_OrbitNumber   = e.orbitNumber()    & H_ORBITCOUNTER_MASK;
      GT_BunchCrossing = e.bunchCrossing()  & H_BX_MASK;
      GT_TriggerType   = e.experimentType() & H_TTYPE_MASK;

    } else {

      // use the most frequent among the ECAL FEDs

      std::map<int,int> ECALDCC_L1A_FreqMap;
      std::map<int,int> ECALDCC_OrbitNumber_FreqMap;
      std::map<int,int> ECALDCC_BunchCrossing_FreqMap;
      std::map<int,int> ECALDCC_TriggerType_FreqMap;

      int ECALDCC_L1A_MostFreqCounts = 0;
      int ECALDCC_OrbitNumber_MostFreqCounts = 0;
      int ECALDCC_BunchCrossing_MostFreqCounts = 0;
      int ECALDCC_TriggerType_MostFreqCounts = 0;

      edm::Handle<EcalRawDataCollection> dcchs;

      if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

        for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

          if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

          int ECALDCC_L1A = dcchItr->getLV1();
          int ECALDCC_OrbitNumber = dcchItr->getOrbit();
          int ECALDCC_BunchCrossing = dcchItr->getBX();
          int ECALDCC_TriggerType = dcchItr->getBasicTriggerType();

          ++ECALDCC_L1A_FreqMap[ECALDCC_L1A];
          ++ECALDCC_OrbitNumber_FreqMap[ECALDCC_OrbitNumber];
          ++ECALDCC_BunchCrossing_FreqMap[ECALDCC_BunchCrossing];
          ++ECALDCC_TriggerType_FreqMap[ECALDCC_TriggerType];

          if ( ECALDCC_L1A_FreqMap[ECALDCC_L1A] > ECALDCC_L1A_MostFreqCounts ) {
            ECALDCC_L1A_MostFreqCounts = ECALDCC_L1A_FreqMap[ECALDCC_L1A];
            ECALDCC_L1A_MostFreqId = ECALDCC_L1A;
          }

          if ( ECALDCC_OrbitNumber_FreqMap[ECALDCC_OrbitNumber] > ECALDCC_OrbitNumber_MostFreqCounts ) {
            ECALDCC_OrbitNumber_MostFreqCounts = ECALDCC_OrbitNumber_FreqMap[ECALDCC_OrbitNumber];
            ECALDCC_OrbitNumber_MostFreqId = ECALDCC_OrbitNumber;
          }

          if ( ECALDCC_BunchCrossing_FreqMap[ECALDCC_BunchCrossing] > ECALDCC_BunchCrossing_MostFreqCounts ) {
            ECALDCC_BunchCrossing_MostFreqCounts = ECALDCC_BunchCrossing_FreqMap[ECALDCC_BunchCrossing];
            ECALDCC_BunchCrossing_MostFreqId = ECALDCC_BunchCrossing;
          }

          if ( ECALDCC_TriggerType_FreqMap[ECALDCC_TriggerType] > ECALDCC_TriggerType_MostFreqCounts ) {
            ECALDCC_TriggerType_MostFreqCounts = ECALDCC_TriggerType_FreqMap[ECALDCC_TriggerType];
            ECALDCC_TriggerType_MostFreqId = ECALDCC_TriggerType;
          }

        }

      } else {
        edm::LogWarning("EBRawDataTask") << EcalRawDataCollection_ << " not available";
      }

    }

    // ECAL barrel FEDs
    int EBFirstFED=610;
    for(int i=0; i<36; i++) {

      const FEDRawData& fedData = allFedRawData->FEDData(EBFirstFED+i);

      int length = fedData.size()/sizeof(uint64_t);

      if ( length > 0 ) {

        uint64_t * pData = (uint64_t *)(fedData.data());
        uint64_t * fedTrailer = pData + (length - 1);
        bool crcError = (*fedTrailer >> 2 ) & 0x1;

        if (crcError) meEBCRCErrors_->Fill( i+1 );

      }

    }

  } else {
    edm::LogWarning("EBRawDataTask") << FEDRawDataCollection_ << " not available";
  }

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );
      float xism = ism+0.5;

      int ECALDCC_runNumber     = dcchItr->getRunNumber();

      int ECALDCC_L1A           = dcchItr->getLV1();
      int ECALDCC_OrbitNumber   = dcchItr->getOrbit();
      int ECALDCC_BunchCrossing = dcchItr->getBX();
      int ECALDCC_TriggerType   = dcchItr->getBasicTriggerType();

      if ( evt_runNumber != ECALDCC_runNumber ) meEBRunNumberErrors_->Fill( xism );

      if ( gtFedDataSize > 0 ) {

        if ( GT_L1A != ECALDCC_L1A ) meEBL1ADCCErrors_->Fill( xism );

        if ( GT_BunchCrossing != ECALDCC_BunchCrossing ) meEBBunchCrossingDCCErrors_->Fill( xism );

        if ( GT_TriggerType != ECALDCC_TriggerType ) meEBTriggerTypeErrors_->Fill ( xism );

      } else {

        if ( ECALDCC_L1A_MostFreqId != ECALDCC_L1A ) meEBL1ADCCErrors_->Fill( xism );

        if ( ECALDCC_BunchCrossing_MostFreqId != ECALDCC_BunchCrossing ) meEBBunchCrossingDCCErrors_->Fill( xism );

        if ( ECALDCC_TriggerType_MostFreqId != ECALDCC_TriggerType ) meEBTriggerTypeErrors_->Fill ( xism );

      }

      if ( gtFedDataSize == 0 ) {

        if ( GT_OrbitNumber != ECALDCC_OrbitNumber ) meEBOrbitNumberErrors_->Fill ( xism );

      } else {

        if ( ECALDCC_OrbitNumber_MostFreqId != ECALDCC_OrbitNumber ) meEBOrbitNumberErrors_->Fill ( xism );

      }

      // DCC vs. FE,TCC, SRP syncronization
      const std::vector<short> feBxs = dcchItr->getFEBxs();
      const std::vector<short> tccBx = dcchItr->getTCCBx();
      const short srpBx = dcchItr->getSRPBx();
      const std::vector<short> status = dcchItr->getFEStatus();

      std::vector<int> BxSynchStatus;
      BxSynchStatus.reserve((int)feBxs.size());

      for(int fe=0; fe<(int)feBxs.size(); fe++) {
        // look for ACTIVE towers only
        if(status[fe] != 0) continue;
        if(feBxs[fe] != ECALDCC_BunchCrossing && feBxs[fe] != -1 && ECALDCC_BunchCrossing != -1) {
          meEBBunchCrossingFEErrors_->Fill( xism, 1/(float)feBxs.size() );
          BxSynchStatus[fe] = 0;
        } else BxSynchStatus[fe] = 1;
      }

      // vector of TCC channels has 4 elements for both EB and EE.
      // EB uses [0], EE uses [0-3].
      if(tccBx.size() == MAX_TCC_SIZE) {
        if(tccBx[0] != ECALDCC_BunchCrossing && tccBx[0] != -1 && ECALDCC_BunchCrossing != -1) meEBBunchCrossingTCCErrors_->Fill( xism );
      }

      if(srpBx != ECALDCC_BunchCrossing && srpBx != -1 && ECALDCC_BunchCrossing != -1) meEBBunchCrossingSRPErrors_->Fill( xism );

      const std::vector<short> feLv1 = dcchItr->getFELv1();
      const std::vector<short> tccLv1 = dcchItr->getTCCLv1();
      const short srpLv1 = dcchItr->getSRPLv1();

      // Lv1 in TCC,SRP,FE are limited to 12 bits(LSB), while in the DCC Lv1 has 24 bits
      int ECALDCC_L1A_12bit = ECALDCC_L1A & 0xfff;
      int feLv1Offset = ( e.isRealData() ) ? 1 : 0; // in MC FE Lv1A counter starts from 1, in data from 0

      for(int fe=0; fe<(int)feLv1.size(); fe++) {
        // look for ACTIVE towers only
        if(status[fe] != 0) continue;
        if(feLv1[fe]+feLv1Offset != ECALDCC_L1A_12bit && feLv1[fe] != -1 && ECALDCC_L1A_12bit - 1 != -1) {
          meEBL1AFEErrors_->Fill( xism, 1/(float)feLv1.size() );
          meEBSynchronizationErrorsByLumi_->Fill( xism, 1/(float)feLv1.size() );
	  errorsInEvent += 1. / feLv1.size();
        } else if( BxSynchStatus[fe]==0 ){
	  meEBSynchronizationErrorsByLumi_->Fill( xism, 1/(float)feLv1.size() );
	  errorsInEvent += 1. / feLv1.size();
	}
      }

      // vector of TCC channels has 4 elements for both EB and EE.
      // EB uses [0], EE uses [0-3].
      if(tccLv1.size() == MAX_TCC_SIZE) {
        if(tccLv1[0] != ECALDCC_L1A_12bit && tccLv1[0] != -1 && ECALDCC_L1A_12bit - 1 != -1) meEBL1ATCCErrors_->Fill( xism );
      }

      if(srpLv1 != ECALDCC_L1A_12bit && srpLv1 != -1 && ECALDCC_L1A_12bit - 1 != -1) meEBL1ASRPErrors_->Fill( xism );

      if ( gtFedDataSize == 0 ) {

        if ( GT_OrbitNumber != ECALDCC_OrbitNumber ) meEBOrbitNumberErrors_->Fill ( xism );

      } else {

        if ( ECALDCC_OrbitNumber_MostFreqId != ECALDCC_OrbitNumber ) meEBOrbitNumberErrors_->Fill ( xism );

      }

      float evtType = dcchItr->getRunType();

      if ( evtType < 0 || evtType > 22 ) evtType = -1;

      if ( ECALDCC_BunchCrossing < calibrationBX_ ) meEBEventTypePreCalibrationBX_->Fill( evtType+0.5, 1./36. );
      if ( ECALDCC_BunchCrossing == calibrationBX_ ) meEBEventTypeCalibrationBX_->Fill( evtType+0.5, 1./36. );
      if ( ECALDCC_BunchCrossing > calibrationBX_ ) meEBEventTypePostCalibrationBX_->Fill ( evtType+0.5, 1./36. );

      if ( ECALDCC_BunchCrossing != calibrationBX_ ) {
        if ( evtType != EcalDCCHeaderBlock::COSMIC &&
             evtType != EcalDCCHeaderBlock::MTCC &&
             evtType != EcalDCCHeaderBlock::COSMICS_GLOBAL &&
             evtType != EcalDCCHeaderBlock::PHYSICS_GLOBAL &&
             evtType != EcalDCCHeaderBlock::COSMICS_LOCAL &&
             evtType != EcalDCCHeaderBlock::PHYSICS_LOCAL &&
             evtType != -1 ) meEBCalibrationEventErrors_->Fill( xism );
      } else {
        if ( evtType == EcalDCCHeaderBlock::COSMIC ||
             evtType == EcalDCCHeaderBlock::MTCC ||
             evtType == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
             evtType == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
             evtType == EcalDCCHeaderBlock::COSMICS_LOCAL ||
             evtType == EcalDCCHeaderBlock::PHYSICS_LOCAL ) meEBCalibrationEventErrors_->Fill( xism );
      }

    }

  } else {
    edm::LogWarning("EBRawDataTask") << EcalRawDataCollection_ << " not available";
  }

  if(errorsInEvent > 0.){
    meEBSynchronizationErrorsTrend_->Fill(ls_ - 0.5, errorsInEvent);
    fatalErrors_ += errorsInEvent;
  }
}

