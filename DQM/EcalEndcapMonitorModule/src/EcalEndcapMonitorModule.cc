/*
 * \file EcalEndcapMonitorModule.cc
 *
 * $Date: 2009/06/23 17:11:24 $
 * $Revision: 1.65 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
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

#include <DQM/EcalEndcapMonitorModule/interface/EcalEndcapMonitorModule.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalEndcapMonitorModule::EcalEndcapMonitorModule(const ParameterSet& ps){

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  if ( verbose_ ) {
    cout << endl;
    cout << " *** Ecal Endcap Generic Monitor ***" << endl;
    cout << endl;
  }

  init_ = false;

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
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

  // this should come from the EcalEndcap event header
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
    LogInfo("EcalEndcapMonitorModule") << " debug switch is ON";
  } else {
    LogInfo("EcalEndcapMonitorModule") << " debug switch is OFF";
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

  meEEDCC_ = 0;

  for (int i = 0; i < 2; i++) {
    meEEdigis_[i] = 0;
    meEEhits_[i] = 0;
    meEEtpdigis_[i] = 0;
  }

  for (int i = 0; i < 18; i++) {
    meEvent_[i] = 0;
  }

}

EcalEndcapMonitorModule::~EcalEndcapMonitorModule(){

}

void EcalEndcapMonitorModule::beginJob(const EventSetup& c){

  if ( debug_ ) cout << "EcalEndcapMonitorModule: beginJob" << endl;

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

void EcalEndcapMonitorModule::beginRun(const Run& r, const EventSetup& c) {

  if ( debug_ ) cout << "EcalEndcapMonitorModule: beginRun" << endl;

  if ( ! mergeRuns_ ) this->reset();
  
}
  
void EcalEndcapMonitorModule::endRun(const Run& r, const EventSetup& c) {

  if ( debug_ ) cout << "EcalEndcapMonitorModule: endRun" << endl;

  // end-of-run
  if ( meStatus_ ) meStatus_->Fill(2);

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

}

void EcalEndcapMonitorModule::reset(void) {

  if ( meEvtType_ ) meEvtType_->Reset();

  if ( meEEDCC_ ) meEEDCC_->Reset();

  for (int i = 0; i < 2; i++) {
    if ( meEEdigis_[i] ) meEEdigis_[i]->Reset();

    if ( meEEhits_[i] ) meEEdigis_[i]->Reset();

    if ( meEEtpdigis_[i] ) meEEtpdigis_[i]->Reset();
  }

  if ( enableEventDisplay_ ) {
    for (int i = 0; i < 18; i++) {
      if ( meEvent_[i] ) meEvent_[i]->Reset();
    }
  }

}

void EcalEndcapMonitorModule::setup(void){

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

    meEEDCC_ = dqmStore_->book1D("EEMM DCC", "EEMM DCC", 18, 1, 19.);
    for (int i = 0; i < 18; i++) {
      meEEDCC_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    meEEdigis_[0] = dqmStore_->book1D("EEMM digi number", "EEMM digi number", 100, 0., 13299.);

    meEEdigis_[1] = dqmStore_->bookProfile("EEMM digi number profile", "EEMM digi number profile", 18, 1, 19., 850, 0., 851., "s");
    for (int i = 0; i < 18; i++) {
      meEEdigis_[1]->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    meEEhits_[0] = dqmStore_->book1D("EEMM hit number", "EEMM hit number", 100, 0., 13299.);

    meEEhits_[1] = dqmStore_->bookProfile("EEMM hit number profile", "EEMM hit number profile", 18, 1, 19., 850, 0., 851., "s");
    for (int i = 0; i < 18; i++) {
      meEEhits_[1]->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    meEEtpdigis_[0] = dqmStore_->book1D("EEMM TP digi number", "EEMM TP digi number", 100, 0., 631.);

    meEEtpdigis_[1] = dqmStore_->bookProfile("EEMM TP digi number profile", "EEMM TP digi number profile", 18, 1, 19., 34, 0., 35., "s");
    for (int i = 0; i < 18; i++) {
      meEEtpdigis_[1]->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    if ( enableEventDisplay_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/EcalEvent");
      for (int i = 0; i < 18; i++) {
        sprintf(histo, "EEMM event %s", Numbers::sEE(i+1).c_str());
        meEvent_[i] = dqmStore_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
        meEvent_[i]->setAxisTitle("jx", 1);
        meEvent_[i]->setAxisTitle("jy", 2);
        dqmStore_->tag(meEvent_[i], i+1);
        if ( meEvent_[i] ) meEvent_[i]->setResetMe(true);
      }
    }

  }

}

void EcalEndcapMonitorModule::cleanup(void){

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

    if ( meEEDCC_ ) dqmStore_->removeElement( meEEDCC_->getName() );
    meEEDCC_ = 0;

    for (int i = 0; i < 2; i++) {

      if ( meEEdigis_[i] ) dqmStore_->removeElement( meEEdigis_[i]->getName() );
      meEEdigis_[i] = 0;

      if ( meEEhits_[i] ) dqmStore_->removeElement( meEEhits_[i]->getName() );
      meEEhits_[i] = 0;

      if ( meEEtpdigis_[i] ) dqmStore_->removeElement( meEEtpdigis_[i]->getName() );
      meEEtpdigis_[i] = 0;

    }

    if ( enableEventDisplay_ ) {

      dqmStore_->setCurrentFolder(prefixME_ + "/EcalEvent");

      for (int i = 0; i < 18; i++) {

        if ( meEvent_[i] ) dqmStore_->removeElement( meEvent_[i]->getName() );
        meEvent_[i] = 0;

      }

    }

  }

  init_ = false;

}

void EcalEndcapMonitorModule::endJob(void) {

  if ( debug_ ) cout << "EcalEndcapMonitorModule: endJob, ievt = " << ievt_ << endl;

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

void EcalEndcapMonitorModule::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c, verbose_);

  if ( ! init_ ) this->setup();

  ievt_++;

  LogInfo("EcalEndcapMonitorModule") << "processing event " << ievt_;

  if ( ! fixedRunNumber_ ) runNumber_ = e.id().run();

  evtNumber_ = e.id().event();

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    if ( dcchs->size() == 0 ) {
      LogWarning("EcalEndcapMonitorModule") << EcalRawDataCollection_ << " is empty";
      return;
    }

    int neec = 0;

    int ndccs = dcchs->size();

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      neec++;

      if ( meEEDCC_ ) meEEDCC_->Fill(Numbers::iSM( *dcchItr, EcalEndcap )+0.5);

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

    LogDebug("EcalEndcapMonitorModule") << "event: " << ievt_ << " DCC headers collection size: " << neec;

  } else {

    if ( evtType_ < 0 || evtType_ > 22 ) evtType_ = -1;
    if ( meEvtType_ ) meEvtType_->Fill(evtType_+0.5, 1./18.);

    LogWarning("EcalEndcapMonitorModule") << EcalRawDataCollection_ << " not available";

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
    LogInfo("EcalEndcapMonitorModule") << "processing run " << runNumber_;
    // begin-of-run
    if ( meStatus_ ) meStatus_->Fill(0);
  } else {
    // running
    if ( meStatus_ ) meStatus_->Fill(1);
  }

  if ( meRun_ ) meRun_->Fill(runNumber_);
  if ( meEvt_ ) meEvt_->Fill(evtNumber_);

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EcalEndcapMonitorModule") << "event " << ievt_ << " digi collection size " << need;

    int counter[18] = { 0 };

    if ( meEEdigis_[0] ) {
      if ( isPhysics_ ) meEEdigis_[0]->Fill(float(need));
    }

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      counter[ism-1]++;

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      LogDebug("EcalEndcapMonitorModule") << " det id = " << id;
      LogDebug("EcalEndcapMonitorModule") << " sm, ix, iy " << ism << " " << ix << " " << iy;

    }

    for (int i = 0; i < 18; i++) {

      if ( meEEdigis_[1] ) {
        if ( isPhysics_ ) meEEdigis_[1]->Fill(i+1+0.5, counter[i]);
      }

    }

  } else {

    LogWarning("EcalEndcapMonitorModule") << EEDigiCollection_ << " not available";

  }

  Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int neeh = hits->size();
    LogDebug("EcalEndcapMonitorModule") << "event " << ievt_ << " hits collection size " << neeh;

    if ( meEEhits_[0] ) { 
      if ( isPhysics_ ) meEEhits_[0]->Fill(float(neeh));
    }

    int counter[18] = { 0 };

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EcalRecHit hit = (*hitItr);
      EEDetId id = hit.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      counter[ism-1]++;

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      LogDebug("EcalEndcapMonitorModule") << " det id = " << id;
      LogDebug("EcalEndcapMonitorModule") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      float xval = hit.energy();

      LogDebug("EcalEndcapMonitorModule") << " hit energy " << xval;

      if ( enableEventDisplay_ ) {

        if ( xval >= 10 ) {
          if ( meEvent_[ism-1] ) meEvent_[ism-1]->Fill(xix, xiy, xval);
        }

      }

    }

    for (int i = 0; i < 18; i++) {

      if ( meEEhits_[1] ) {
        if ( isPhysics_ ) meEEhits_[1]->Fill(i+1+0.5, counter[i]);
      }

    }

  } else {

    LogWarning("EcalEndcapMonitorModule") << EcalRecHitCollection_ << " not available";

  }

  Handle<EcalTrigPrimDigiCollection> tpdigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, tpdigis) ) {

    int neetpd = 0;
    int counter[18] = { 0 };

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = tpdigis->begin();
          tpdigiItr != tpdigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId idt = data.id();

      if ( Numbers::subDet( idt ) != EcalEndcap ) continue;

      int ismt = Numbers::iSM( idt );

      neetpd++;
      counter[ismt-1]++;

    }

    LogDebug("EcalEndcapMonitorModule") << "event " << ievt_ << " TP digi collection size " << neetpd;
    if ( meEEtpdigis_[0] ) {
      if ( isPhysics_ ) meEEtpdigis_[0]->Fill(float(neetpd));
    }

    for (int i = 0; i < 18; i++) {

      if ( meEEtpdigis_[1] ) {
        if ( isPhysics_ ) meEEtpdigis_[1]->Fill(i+1+0.5, counter[i]);
      }

    }

  } else {

    LogWarning("EcalEndcapMonitorModule") << EcalTrigPrimDigiCollection_ << " not available";

  }

}

