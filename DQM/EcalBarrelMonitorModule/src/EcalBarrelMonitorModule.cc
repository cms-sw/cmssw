/*
 * \file EcalBarrelMonitorModule.cc
 *
 * $Date: 2009/06/23 17:11:24 $
 * $Revision: 1.188 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const ParameterSet& ps){

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if ( verbose_ ) {
    cout << endl;
    cout << " *** Ecal Barrel Generic Monitor ***" << endl;
    cout << endl;
  }

  init_ = false;

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  // this should come from the event header
  runNumber_ = ps.getUntrackedParameter<int>("runNumber", 0);

  fixedRunNumber_ = false;
  if ( runNumber_ != 0 ) fixedRunNumber_ = true;

  if ( fixedRunNumber_ ) {
    LogInfo("EcalBarrelMonitorModule") << " using fixed Run Number = " << runNumber_ << endl;
  }

  // this should come from the event header
  evtNumber_ = 0;

  // this should come from the EcalBarrel event header
  runType_ = ps.getUntrackedParameter<int>("runType", -1);
  evtType_ = runType_;

  fixedRunType_ = false;
  if ( runType_ != -1 ) fixedRunType_ = true;

  if ( fixedRunType_) {
    LogInfo("EcalBarrelMonitorModule") << " using fixed Run Type = " << runType_ << endl;
  }

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  if ( debug_ ) {
    LogInfo("EcalBarrelMonitorModule") << " debug switch is ON";
  } else {
    LogInfo("EcalBarrelMonitorModule") << " debug switch is OFF";
  }

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // enableCleanup switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // mergeRuns switch
  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  if ( enableCleanup_ ) {
    LogInfo("EcalBarrelMonitorModule") << " enableCleanup switch is ON";
  } else {
    LogInfo("EcalBarrelMonitorModule") << " enableCleanup switch is OFF";
  }

  // EventDisplay switch
  enableEventDisplay_ = ps.getUntrackedParameter<bool>("enableEventDisplay", false);

  meStatus_ = 0;
  meRun_ = 0;
  meEvt_ = 0;
  meRunType_ = 0;
  meEvtType_ = 0;

  meEBDCC_ = 0;

  for (int i = 0; i < 2; i++) {
    meEBdigis_[i] = 0;
    meEBhits_[i] = 0;
    meEBtpdigis_[i] = 0;
  }

  for (int i = 0; i < 36; i++) {
    meEvent_[i] = 0;
  }

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

}

void EcalBarrelMonitorModule::beginJob(const EventSetup& c){

  if ( debug_ ) cout << "EcalBarrelMonitorModule: beginJob" << endl;

  ievt_ = 0;

  dqmStore_ = Service<DQMStore>().operator->();

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");
    dqmStore_->rmdir(prefixME_ + "/EcalInfo");
    if ( enableEventDisplay_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EcalEvent");
      dqmStore_->rmdir(prefixME_ + "/EcalEvent");
    }
  }

}

void EcalBarrelMonitorModule::beginRun(const Run& r, const EventSetup& c) {

  if ( debug_ ) cout << "EcalBarrelMonitorModule: beginRun" << endl;

  if ( ! mergeRuns_ ) this->reset();

}

void EcalBarrelMonitorModule::endRun(const Run& r, const EventSetup& c) {

  if ( debug_ ) cout << "EcalBarrelMonitorModule: endRun" << endl;

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

}

void EcalBarrelMonitorModule::reset(void) {

  if ( meEvtType_ ) meEvtType_->Reset();

  if ( meEBDCC_ ) meEBDCC_->Reset();
    
  for (int i = 0; i < 2; i++) {
    if ( meEBdigis_[i] ) meEBdigis_[i]->Reset();
    
    if ( meEBhits_[i] ) meEBdigis_[i]->Reset();
    
    if ( meEBtpdigis_[i] ) meEBtpdigis_[i]->Reset();
  } 

  if ( enableEventDisplay_ ) {
    for (int i = 0; i < 18; i++) {
      if ( meEvent_[i] ) meEvent_[i]->Reset();
    }
  } 

}

void EcalBarrelMonitorModule::setup(void){

  init_ = true;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");

    meStatus_ = dqmStore_->bookInt("STATUS");

    meRun_ = dqmStore_->bookInt("RUN");
    meEvt_ = dqmStore_->bookInt("EVT");

    meRunType_ = dqmStore_->bookInt("RUNTYPE");
    meEvtType_ = dqmStore_->book1D("EVTTYPE", "EVTTYPE", 31, -1., 30.);
    meEvtType_->setAxisTitle("number of events", 2);
    meEvtType_->setBinLabel(1, "UNKNOWN", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::COSMIC, "COSMIC", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH4, "BEAMH4", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::BEAMH2, "BEAMH2", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::MTCC, "MTCC", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::LASER_STD, "LASER_STD", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::LASER_POWER_SCAN, "LASER_POWER_SCAN", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::LASER_DELAY_SCAN, "LASER_DELAY_SCAN", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_SCAN_MEM, "TESTPULSE_SCAN_MEM", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_MGPA, "TESTPULSE_MGPA", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_STD, "PEDESTAL_STD", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN, "PEDESTAL_OFFSET_SCAN", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_25NS_SCAN, "PEDESTAL_25NS_SCAN", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::LED_STD, "LED_STD", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_GLOBAL, "PHYSICS_GLOBAL", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_GLOBAL, "COSMICS_GLOBAL", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::HALO_GLOBAL, "HALO_GLOBAL", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::LASER_GAP, "LASER_GAP", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::TESTPULSE_GAP, "TESTPULSE_GAP");
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::PEDESTAL_GAP, "PEDESTAL_GAP");
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::LED_GAP, "LED_GAP", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::PHYSICS_LOCAL, "PHYSICS_LOCAL", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::COSMICS_LOCAL, "COSMICS_LOCAL", 1);
    meEvtType_->setBinLabel(2+EcalDCCHeaderBlock::HALO_LOCAL, "HALO_LOCAL", 1);
  }

  // unknown
  if ( meStatus_ ) meStatus_->Fill(-1);

  if ( meRun_ ) meRun_->Fill(-1);
  if ( meEvt_ ) meEvt_->Fill(-1);

  if ( meRunType_ ) meRunType_->Fill(-1);

  char histo[20];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");

    meEBDCC_ = dqmStore_->book1D("EBMM DCC", "EBMM DCC", 36, 1, 37.);
    for (int i = 0; i < 36; i++) {
      meEBDCC_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    meEBdigis_[0] = dqmStore_->book1D("EBMM digi number", "EBMM digi number", 100, 0., 61201.);

    meEBdigis_[1] = dqmStore_->bookProfile("EBMM digi number profile", "EBMM digi number profile", 36, 1, 37., 1700, 0., 1701., "s");
    for (int i = 0; i < 36; i++) {
      meEBdigis_[1]->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    meEBhits_[0] = dqmStore_->book1D("EBMM hit number", "EBMM hit number", 100, 0., 61201.);

    meEBhits_[1] = dqmStore_->bookProfile("EBMM hit number profile", "EBMM hit number profile", 36, 1, 37., 1700, 0., 1701., "s");
    for (int i = 0; i < 36; i++) {
      meEBhits_[1]->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    meEBtpdigis_[0] = dqmStore_->book1D("EBMM TP digi number", "EBMM TP digi number", 100, 0., 2449.);

    meEBtpdigis_[1] = dqmStore_->bookProfile("EBMM TP digi number profile", "EBMM TP digi number profile", 36, 1, 37., 68, 0., 69., "s");
    for (int i = 0; i < 36; i++) {
      meEBtpdigis_[1]->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    if ( enableEventDisplay_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EcalEvent");
      for (int i = 0; i < 36; i++) {
        sprintf(histo, "EBMM event %s", Numbers::sEB(i+1).c_str());
        meEvent_[i] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
        meEvent_[i]->setAxisTitle("ieta", 1);
        meEvent_[i]->setAxisTitle("iphi", 2);
        dqmStore_->tag(meEvent_[i], i+1);
        if ( meEvent_[i] ) meEvent_[i]->setResetMe(true);
      }
    }

  }

}

void EcalBarrelMonitorModule::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EcalInfo");

    if ( meStatus_ ) dqmStore_->removeElement( meStatus_->getName() );
    meStatus_ = 0;

    if ( meRun_ ) dqmStore_->removeElement( meRun_->getName() );
    meRun_ = 0;

    if ( meEvt_ ) dqmStore_->removeElement( meEvt_->getName() );
    meEvt_ = 0;

    if ( meRunType_ ) dqmStore_->removeElement( meRunType_->getName() );
    meRunType_ = 0;

    if ( meEvtType_ ) dqmStore_->removeElement( meEvtType_->getName() );
    meEvtType_ = 0;

    if ( meEBDCC_ ) dqmStore_->removeElement( meEBDCC_->getName() );
    meEBDCC_ = 0;

    for (int i = 0; i < 2; i++) {

      if ( meEBdigis_[i] ) dqmStore_->removeElement( meEBdigis_[i]->getName() );
      meEBdigis_[i] = 0;

      if ( meEBhits_[i] ) dqmStore_->removeElement( meEBhits_[i]->getName() );
      meEBhits_[i] = 0;

      if ( meEBtpdigis_[i] ) dqmStore_->removeElement( meEBtpdigis_[i]->getName() );
      meEBtpdigis_[i] = 0;

    }

    if ( enableEventDisplay_ ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EcalEvent");

      for (int i = 0; i < 36; i++) {

        if ( meEvent_[i] ) dqmStore_->removeElement( meEvent_[i]->getName() );
        meEvent_[i] = 0;

      }

    }

  }

  init_ = false;

}

void EcalBarrelMonitorModule::endJob(void) {

  if ( debug_ ) cout << "EcalBarrelMonitorModule: endJob, ievt = " << ievt_ << endl;

  if ( dqmStore_ ) {
    meStatus_ = dqmStore_->get(prefixME_ + "/EventInfo/STATUS");
    meRun_ = dqmStore_->get(prefixME_ + "/EventInfo/RUN");
    meEvt_ = dqmStore_->get(prefixME_ + "/EventInfo/EVT");
  }

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

  if ( init_ ) this->cleanup();

}

void EcalBarrelMonitorModule::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c, verbose_);

  if ( ! init_ ) this->setup();

  ievt_++;

  LogInfo("EcalBarrelMonitorModule") << "processing event " << ievt_;

  if ( ! fixedRunNumber_ ) runNumber_ = e.id().run();

  evtNumber_ = e.id().event();

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    if ( dcchs->size() == 0 ) {
      LogWarning("EcalBarrelMonitorModule") << EcalRawDataCollection_ << " is empty";
      return;
    }

    int nebc = 0;

    int ndccs = dcchs->size();

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      nebc++;

      if ( meEBDCC_ ) meEBDCC_->Fill(Numbers::iSM( *dcchItr, EcalBarrel )+0.5);

      if ( ! fixedRunNumber_ ) {
        runNumber_ = dcchItr->getRunNumber();
      }

      evtNumber_ = dcchItr->getLV1();

      if ( ! fixedRunType_ ) {
        runType_ = dcchItr->getRunType();
        evtType_ = runType_;
      }

      if ( evtType_ < 0 || evtType_ > 22 ) evtType_ = -1;
      if ( meEvtType_ ) meEvtType_->Fill(evtType_+0.5, 1./ndccs);
                
    }

    LogDebug("EcalBarrelMonitorModule") << "event: " << ievt_ << " DCC headers collection size: " << nebc;

  } else {

    if ( evtType_ < 0 || evtType_ > 22 ) evtType_ = -1;
    if ( meEvtType_ ) meEvtType_->Fill(evtType_+0.5, 1./36.);

    LogWarning("EcalBarrelMonitorModule") << EcalRawDataCollection_ << " not available";

  }

  isPhysics_ = false;
  if ( evtType_ == EcalDCCHeaderBlock::COSMIC ||
       evtType_ == EcalDCCHeaderBlock::MTCC ||
       evtType_ == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
       evtType_ == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
       evtType_ == EcalDCCHeaderBlock::COSMICS_LOCAL ||
       evtType_ == EcalDCCHeaderBlock::PHYSICS_LOCAL ) isPhysics_ = true;

  if ( meRunType_ ) meRunType_->Fill(runType_);

  if ( ievt_ == 1 ) {
    LogInfo("EcalBarrelMonitorModule") << "processing run " << runNumber_;
    // begin-of-run
    if ( meStatus_ ) meStatus_->Fill(0);
  } else {
    // running
    if ( meStatus_ ) meStatus_->Fill(1);
  }

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

  Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int counter[36] = { 0 };

    int nebd = digis->size();
    LogDebug("EcalBarrelMonitorModule") << "event " << ievt_ << " digi collection size " << nebd;

    if ( meEBdigis_[0] ) { 
      if ( isPhysics_ ) meEBdigis_[0]->Fill(float(nebd));
    }

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDataFrame dataframe = (*digiItr);
      EBDetId id = dataframe.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      counter[ism-1]++;

      LogDebug("EcalBarrelMonitorModule") << " det id = " << id;
      LogDebug("EcalBarrelMonitorModule") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

    }

    for (int i = 0; i < 36; i++) {

      if ( meEBdigis_[1] ) {
        if ( isPhysics_ ) meEBdigis_[1]->Fill(i+1+0.5, counter[i]);
      }
    }

  } else {

    LogWarning("EcalBarrelMonitorModule") << EBDigiCollection_ << " not available";

  }

  Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int nebh = hits->size();
    LogDebug("EcalBarrelMonitorModule") << "event " << ievt_ << " hits collection size " << nebh;

    if ( meEBhits_[0] ) {
      if ( isPhysics_ ) meEBhits_[0]->Fill(float(nebh));
    }

    int counter[36] = { 0 };

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalRecHit hit = (*hitItr);
      EBDetId id = hit.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      counter[ism-1]++;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EcalBarrelMonitorModule") << " det id = " << id;
      LogDebug("EcalBarrelMonitorModule") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

      float xval = hit.energy();

      LogDebug("EcalBarrelMonitorModule") << " hit energy " << xval;

      if ( enableEventDisplay_ ) {

        if ( xval >= 10 ) {
          if ( meEvent_[ism-1] ) meEvent_[ism-1]->Fill(xie, xip, xval);
        }

      }

    }

    for (int i = 0; i < 36; i++) {

      if ( meEBhits_[1] ) {
        if ( isPhysics_ ) meEBhits_[1]->Fill(i+1+0.5, counter[i]);
      }

    }

  } else {

    LogWarning("EcalBarrelMonitorModule") << EcalRecHitCollection_ << " not available";

  }

  Handle<EcalTrigPrimDigiCollection> tpdigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, tpdigis) ) {

    int nebtpd = 0;
    int counter[36] = { 0 };

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = tpdigis->begin();
          tpdigiItr != tpdigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId idt = data.id();

      if ( Numbers::subDet( idt ) != EcalBarrel ) continue;

      int ismt = Numbers::iSM( idt );

      nebtpd++;
      counter[ismt-1]++;

    }

    LogDebug("EcalBarrelMonitorModule") << "event " << ievt_ << " TP digi collection size " << nebtpd;
    if ( meEBtpdigis_[0] ) {
      if ( isPhysics_ ) meEBtpdigis_[0]->Fill(float(nebtpd));
    }

    for (int i = 0; i < 36; i++) {

      if ( meEBtpdigis_[1] ) {
        if ( isPhysics_ ) meEBtpdigis_[1]->Fill(i+1+0.5, counter[i]);
      }

    }

  } else {

    LogWarning("EcalBarrelMonitorModule") << EcalTrigPrimDigiCollection_ << " not available";

  }

}

