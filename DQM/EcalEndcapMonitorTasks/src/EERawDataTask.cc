/*
 * \file EERawDataTask.cc
 *
 * $Date: 2012/06/28 12:14:30 $
 * $Revision: 1.45 $
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

#include "DQM/EcalEndcapMonitorTasks/interface/EERawDataTask.h"

EERawDataTask::EERawDataTask(const edm::ParameterSet& ps) {

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  meEEEventTypePreCalibrationBX_ = 0;
  meEEEventTypeCalibrationBX_ = 0;
  meEEEventTypePostCalibrationBX_ = 0;
  meEECRCErrors_ = 0;
  meEERunNumberErrors_ = 0;
  meEEOrbitNumberErrors_ = 0;
  meEETriggerTypeErrors_ = 0;
  meEECalibrationEventErrors_ = 0;
  meEEL1ADCCErrors_ = 0;
  meEEBunchCrossingDCCErrors_ = 0;
  meEEL1AFEErrors_ = 0;
  meEEBunchCrossingFEErrors_ = 0;
  meEEL1ATCCErrors_ = 0;
  meEEBunchCrossingTCCErrors_ = 0;
  meEEL1ASRPErrors_ = 0;
  meEEBunchCrossingSRPErrors_ = 0;

  meEESynchronizationErrorsByLumi_ = 0;

  meEESynchronizationErrorsTrend_ = 0;

  calibrationBX_ = 3490;

}

EERawDataTask::~EERawDataTask() {
}

void EERawDataTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EERawDataTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EERawDataTask/" + subfolder_);
    dqmStore_->rmdir(prefixME_ + "/EERawDataTask");
  }

}

void EERawDataTask::beginLuminosityBlock(const edm::LuminosityBlock& _lumi, const  edm::EventSetup&) {

  if ( ! init_ ) this->setup();

  ls_ = _lumi.luminosityBlock();

  if ( meEESynchronizationErrorsByLumi_ ) meEESynchronizationErrorsByLumi_->Reset();

  if ( meEESynchronizationErrorsTrend_ ){
    int bin(meEESynchronizationErrorsTrend_->getTH1()->GetXaxis()->FindBin(ls_ - 0.5));
    meEESynchronizationErrorsTrend_->setBinContent(bin - 1, fatalErrors_);
    meEESynchronizationErrorsTrend_->getTH1()->SetEntries(fatalErrors_);
  }

}

void EERawDataTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  fatalErrors_ = 0.;

  if ( meEESynchronizationErrorsTrend_ ){
    meEESynchronizationErrorsTrend_->getTH1()->GetXaxis()->SetLimits(0., 50.);
  }
}

void EERawDataTask::endRun(const edm::Run& r, const edm::EventSetup& c) {
}

void EERawDataTask::reset(void) {

  if ( meEEEventTypePreCalibrationBX_ ) meEEEventTypePreCalibrationBX_->Reset();
  if ( meEEEventTypeCalibrationBX_ ) meEEEventTypeCalibrationBX_->Reset();
  if ( meEEEventTypePostCalibrationBX_ ) meEEEventTypePostCalibrationBX_->Reset();
  if ( meEECRCErrors_ ) meEECRCErrors_->Reset();
  if ( meEERunNumberErrors_ ) meEERunNumberErrors_->Reset();
  if ( meEEOrbitNumberErrors_ ) meEEOrbitNumberErrors_->Reset();
  if ( meEETriggerTypeErrors_ ) meEETriggerTypeErrors_->Reset();
  if ( meEECalibrationEventErrors_ ) meEECalibrationEventErrors_->Reset();
  if ( meEEL1ADCCErrors_ ) meEEL1ADCCErrors_->Reset();
  if ( meEEBunchCrossingDCCErrors_ ) meEEBunchCrossingDCCErrors_->Reset();
  if ( meEEL1AFEErrors_ ) meEEL1AFEErrors_->Reset();
  if ( meEEBunchCrossingFEErrors_ ) meEEBunchCrossingFEErrors_->Reset();
  if ( meEEL1ATCCErrors_ ) meEEL1ATCCErrors_->Reset();
  if ( meEEBunchCrossingTCCErrors_ ) meEEBunchCrossingTCCErrors_->Reset();
  if ( meEEL1ASRPErrors_ ) meEEL1ASRPErrors_->Reset();
  if ( meEEBunchCrossingSRPErrors_ ) meEEBunchCrossingSRPErrors_->Reset();
  if ( meEESynchronizationErrorsByLumi_ ) meEESynchronizationErrorsByLumi_->Reset();
  if ( meEESynchronizationErrorsTrend_ ) meEESynchronizationErrorsTrend_->Reset();
}

void EERawDataTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EERawDataTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EERawDataTask/" + subfolder_);

    name = "EERDT event type pre calibration BX";
    meEEEventTypePreCalibrationBX_ = dqmStore_->book1D(name, name, 31, -1., 30.);
    meEEEventTypePreCalibrationBX_->setBinLabel(1, "UNKNOWN", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEEEventTypePreCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);

    name = "EERDT event type calibration BX";
    meEEEventTypeCalibrationBX_ = dqmStore_->book1D(name, name, 31, -1., 30.);
    meEEEventTypeCalibrationBX_->setBinLabel(1, "UNKNOWN", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEEEventTypeCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);

    name = "EERDT event type post calibration BX";
    meEEEventTypePostCalibrationBX_ = dqmStore_->book1D(name, name, 31, -1., 30.);
    meEEEventTypePostCalibrationBX_->setBinLabel(1, "UNKNOWN", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEEEventTypePostCalibrationBX_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);

    name = "EERDT CRC errors";
    meEECRCErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEECRCErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT run number errors";
    meEERunNumberErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEERunNumberErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT orbit number errors";
    meEEOrbitNumberErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEOrbitNumberErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT trigger type errors";
    meEETriggerTypeErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEETriggerTypeErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT calibration event errors";
    meEECalibrationEventErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEECalibrationEventErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT L1A DCC errors";
    meEEL1ADCCErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEL1ADCCErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT bunch crossing DCC errors";
    meEEBunchCrossingDCCErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEBunchCrossingDCCErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT L1A FE errors";
    meEEL1AFEErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEL1AFEErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT bunch crossing FE errors";
    meEEBunchCrossingFEErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEBunchCrossingFEErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT L1A TCC errors";
    meEEL1ATCCErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEL1ATCCErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT bunch crossing TCC errors";
    meEEBunchCrossingTCCErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEBunchCrossingTCCErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT L1A SRP errors";
    meEEL1ASRPErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEL1ASRPErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT bunch crossing SRP errors";
    meEEBunchCrossingSRPErrors_ = dqmStore_->book1D(name, name, 18, 1, 19);
    for (int i = 0; i < 18; i++) {
      meEEBunchCrossingSRPErrors_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT FE synchronization errors by lumi";
    meEESynchronizationErrorsByLumi_ = dqmStore_->book1D(name, name, 18, 1, 19);
    meEESynchronizationErrorsByLumi_->setLumiFlag();
    for (int i = 0; i < 18; i++) {
      meEESynchronizationErrorsByLumi_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EERDT accumulated FE synchronization errors";
    meEESynchronizationErrorsTrend_ = dqmStore_->book1D(name, name, 50, 0., 50.);
    meEESynchronizationErrorsTrend_->setAxisTitle("LumiSection", 1);

  }

}

void EERawDataTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EERawDataTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EERawDataTask/" + subfolder_);

    if ( meEEEventTypePreCalibrationBX_ ) dqmStore_->removeElement( meEEEventTypePreCalibrationBX_->getName() );
    meEEEventTypePreCalibrationBX_ = 0;

    if ( meEEEventTypeCalibrationBX_ ) dqmStore_->removeElement( meEEEventTypeCalibrationBX_->getName() );
    meEEEventTypeCalibrationBX_ = 0;

    if ( meEEEventTypePostCalibrationBX_ ) dqmStore_->removeElement( meEEEventTypePostCalibrationBX_->getName() );
    meEEEventTypePostCalibrationBX_ = 0;

    if ( meEECRCErrors_ ) dqmStore_->removeElement( meEECRCErrors_->getName() );
    meEECRCErrors_ = 0;

    if ( meEERunNumberErrors_ ) dqmStore_->removeElement( meEERunNumberErrors_->getName() );
    meEERunNumberErrors_ = 0;

    if ( meEEOrbitNumberErrors_ ) dqmStore_->removeElement( meEEOrbitNumberErrors_->getName() );
    meEEOrbitNumberErrors_ = 0;

    if ( meEETriggerTypeErrors_ ) dqmStore_->removeElement( meEETriggerTypeErrors_->getName() );
    meEETriggerTypeErrors_ = 0;

    if ( meEECalibrationEventErrors_ ) dqmStore_->removeElement( meEECalibrationEventErrors_->getName() );
    meEECalibrationEventErrors_ = 0;

    if ( meEEL1ADCCErrors_ ) dqmStore_->removeElement( meEEL1ADCCErrors_->getName() );
    meEEL1ADCCErrors_ = 0;

    if ( meEEBunchCrossingDCCErrors_ ) dqmStore_->removeElement( meEEBunchCrossingDCCErrors_->getName() );
    meEEBunchCrossingDCCErrors_ = 0;

    if ( meEEL1AFEErrors_ ) dqmStore_->removeElement( meEEL1AFEErrors_->getName() );
    meEEL1AFEErrors_ = 0;

    if ( meEEBunchCrossingFEErrors_ ) dqmStore_->removeElement( meEEBunchCrossingFEErrors_->getName() );
    meEEBunchCrossingFEErrors_ = 0;

    if ( meEEL1ATCCErrors_ ) dqmStore_->removeElement( meEEL1ATCCErrors_->getName() );
    meEEL1ATCCErrors_ = 0;

    if ( meEEBunchCrossingTCCErrors_ ) dqmStore_->removeElement( meEEBunchCrossingTCCErrors_->getName() );
    meEEBunchCrossingTCCErrors_ = 0;

    if ( meEEL1ASRPErrors_ ) dqmStore_->removeElement( meEEL1ASRPErrors_->getName() );
    meEEL1ASRPErrors_ = 0;

    if ( meEEBunchCrossingSRPErrors_ ) dqmStore_->removeElement( meEEBunchCrossingSRPErrors_->getName() );
    meEEBunchCrossingSRPErrors_ = 0;

    if ( meEESynchronizationErrorsByLumi_ ) dqmStore_->removeElement( meEESynchronizationErrorsByLumi_->getName() );
    meEESynchronizationErrorsByLumi_ = 0;

    if ( meEESynchronizationErrorsTrend_ ) dqmStore_->removeElement( meEESynchronizationErrorsTrend_->getName() );
    meEESynchronizationErrorsTrend_ = 0;
  }

  init_ = false;

}

void EERawDataTask::endLuminosityBlock(const edm::LuminosityBlock& , const  edm::EventSetup&) {
  MonitorElement* me(meEESynchronizationErrorsTrend_);
  if(!me) return;
  if(ls_ >= 50){
    for(int ix(1); ix <= 50; ix++)
      me->setBinContent(ix, me->getBinContent(ix + 1));

    me->getTH1()->GetXaxis()->SetLimits(ls_ - 49, ls_ + 1);
  }
}

void EERawDataTask::endJob(void) {

  edm::LogInfo("EERawDataTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EERawDataTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // fill bin 0 with number of events in the lumi
  if ( meEESynchronizationErrorsByLumi_ ) meEESynchronizationErrorsByLumi_->Fill(0.);

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

          if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

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
        edm::LogWarning("EERawDataTask") << EcalRawDataCollection_ << " not available";
      }

    }

    // ECAL endcap FEDs
    int EEFirstFED[2];
    EEFirstFED[0] = 601; // EE-
    EEFirstFED[1] = 646; // EE+
    for(int zside=0; zside<2; zside++) {

      int firstFedOnSide=EEFirstFED[zside];

      for(int i=0; i<9; i++) {

        const FEDRawData& fedData = allFedRawData->FEDData(firstFedOnSide+i);

        int length = fedData.size()/sizeof(uint64_t);

        if ( length > 0 ) {

          uint64_t * pData = (uint64_t *)(fedData.data());
          uint64_t * fedTrailer = pData + (length - 1);
          bool crcError = (*fedTrailer >> 2 ) & 0x1;

          if (crcError) meEECRCErrors_->Fill( i+1 );

        }

      }

    }

  } else {
    edm::LogWarning("EERawDataTask") << FEDRawDataCollection_ << " not available";
  }

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );
      float xism = ism+0.5;

      int ECALDCC_runNumber     = dcchItr->getRunNumber();

      int ECALDCC_L1A           = dcchItr->getLV1();
      int ECALDCC_OrbitNumber   = dcchItr->getOrbit();
      int ECALDCC_BunchCrossing = dcchItr->getBX();
      int ECALDCC_TriggerType   = dcchItr->getBasicTriggerType();

      if ( evt_runNumber != ECALDCC_runNumber ) meEERunNumberErrors_->Fill( xism );

      if ( gtFedDataSize > 0 ) {

        if ( GT_L1A != ECALDCC_L1A ) meEEL1ADCCErrors_->Fill( xism );

        if ( GT_BunchCrossing != ECALDCC_BunchCrossing ) meEEBunchCrossingDCCErrors_->Fill( xism );

        if ( GT_TriggerType != ECALDCC_TriggerType ) meEETriggerTypeErrors_->Fill ( xism );

      } else {

        if ( ECALDCC_L1A_MostFreqId != ECALDCC_L1A ) meEEL1ADCCErrors_->Fill( xism );

        if ( ECALDCC_BunchCrossing_MostFreqId != ECALDCC_BunchCrossing ) meEEBunchCrossingDCCErrors_->Fill( xism );

        if ( ECALDCC_TriggerType_MostFreqId != ECALDCC_TriggerType ) meEETriggerTypeErrors_->Fill ( xism );

      }

      if ( gtFedDataSize > 0 ) {

        if ( GT_OrbitNumber != ECALDCC_OrbitNumber ) meEEOrbitNumberErrors_->Fill ( xism );

      } else {

        if ( ECALDCC_OrbitNumber_MostFreqId != ECALDCC_OrbitNumber ) meEEOrbitNumberErrors_->Fill ( xism );

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
          meEEBunchCrossingFEErrors_->Fill( xism, 1/(float)feBxs.size());
          BxSynchStatus[fe] = 0;
        } else BxSynchStatus[fe] = 1;
      }

      // vector of TCC channels has 4 elements for both EB and EE.
      // EB uses [0], EE uses [0-3].
      if(tccBx.size() == MAX_TCC_SIZE) {
        for(int tcc=0; tcc<MAX_TCC_SIZE; tcc++) {
          if(tccBx[tcc] != ECALDCC_BunchCrossing && tccBx[tcc] != -1 && ECALDCC_BunchCrossing != -1) meEEBunchCrossingTCCErrors_->Fill( xism, 1/(float)tccBx.size());
        }
      }

      if(srpBx != ECALDCC_BunchCrossing && srpBx != -1 && ECALDCC_BunchCrossing != -1) meEEBunchCrossingSRPErrors_->Fill( xism );

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
          meEEL1AFEErrors_->Fill( xism, 1/(float)feLv1.size());
          meEESynchronizationErrorsByLumi_->Fill( xism, 1/(float)feLv1.size() );
	  errorsInEvent += 1. / feLv1.size();
        } else if( BxSynchStatus[fe]==0 ){
	  meEESynchronizationErrorsByLumi_->Fill( xism, 1/(float)feLv1.size() );
	  errorsInEvent += 1. / feLv1.size();
	}
      }

      // vector of TCC channels has 4 elements for both EB and EE.
      // EB uses [0], EE uses [0-3].
      if(tccLv1.size() == MAX_TCC_SIZE) {
        for(int tcc=0; tcc<MAX_TCC_SIZE; tcc++) {
          if(tccLv1[tcc] != ECALDCC_L1A_12bit && tccLv1[tcc] != -1 && ECALDCC_L1A_12bit - 1 != -1) meEEL1ATCCErrors_->Fill( xism, 1/(float)tccLv1.size());
        }
      }

      if(srpLv1 != ECALDCC_L1A_12bit && srpLv1 != -1 && ECALDCC_L1A_12bit - 1 != -1) meEEL1ASRPErrors_->Fill( xism );

      if ( gtFedDataSize > 0 ) {

        if ( GT_OrbitNumber != ECALDCC_OrbitNumber ) meEEOrbitNumberErrors_->Fill ( xism );

      } else {

        if ( ECALDCC_OrbitNumber_MostFreqId != ECALDCC_OrbitNumber ) meEEOrbitNumberErrors_->Fill ( xism );

      }

      float evtType = dcchItr->getRunType();

      if ( evtType < 0 || evtType > 22 ) evtType = -1;

      if ( ECALDCC_BunchCrossing < calibrationBX_ ) meEEEventTypePreCalibrationBX_->Fill( evtType+0.5, 1./18. );
      if ( ECALDCC_BunchCrossing == calibrationBX_ ) meEEEventTypeCalibrationBX_->Fill( evtType+0.5, 1./18. );
      if ( ECALDCC_BunchCrossing > calibrationBX_ ) meEEEventTypePostCalibrationBX_->Fill ( evtType+0.5, 1./18. );

      if ( ECALDCC_BunchCrossing != calibrationBX_ ) {
        if ( evtType != EcalDCCHeaderBlock::COSMIC &&
             evtType != EcalDCCHeaderBlock::MTCC &&
             evtType != EcalDCCHeaderBlock::COSMICS_GLOBAL &&
             evtType != EcalDCCHeaderBlock::PHYSICS_GLOBAL &&
             evtType != EcalDCCHeaderBlock::COSMICS_LOCAL &&
             evtType != EcalDCCHeaderBlock::PHYSICS_LOCAL &&
             evtType != -1 ) meEECalibrationEventErrors_->Fill( xism );
      } else {
        if ( evtType == EcalDCCHeaderBlock::COSMIC ||
             evtType == EcalDCCHeaderBlock::MTCC ||
             evtType == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
             evtType == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
             evtType == EcalDCCHeaderBlock::COSMICS_LOCAL ||
             evtType == EcalDCCHeaderBlock::PHYSICS_LOCAL ) meEECalibrationEventErrors_->Fill( xism );
      }

    }

  } else {
    edm::LogWarning("EERawDataTask") << EcalRawDataCollection_ << " not available";
  }

  if(errorsInEvent > 0.){
    meEESynchronizationErrorsTrend_->Fill(ls_ - 0.5, errorsInEvent);
    fatalErrors_ += errorsInEvent;
  }
}

