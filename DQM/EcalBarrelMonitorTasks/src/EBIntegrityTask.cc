/*
 * \file EBIntegrityTask.cc
 *
 * $Date: 2007/05/21 09:57:44 $
 * $Revision: 1.39 $
 * \author G. Della Ricca
 *
 */

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBIntegrityTask::EBIntegrityTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  EBDetIdCollection0_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection0");
  EBDetIdCollection1_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection1");
  EBDetIdCollection2_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection2");
  EBDetIdCollection3_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection3");
  EBDetIdCollection4_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection4");
  EcalTrigTowerDetIdCollection1_ = ps.getParameter<edm::InputTag>("EcalTrigTowerDetIdCollection1");
  EcalTrigTowerDetIdCollection2_ = ps.getParameter<edm::InputTag>("EcalTrigTowerDetIdCollection2");
  EcalElectronicsIdCollection1_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection1");
  EcalElectronicsIdCollection2_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection2");
  EcalElectronicsIdCollection3_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection3");
  EcalElectronicsIdCollection4_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection4");

  meIntegrityDCCSize = 0;
  for (int i = 0; i < 36 ; i++) {
    meIntegrityGain[i] = 0;
    meIntegrityChId[i] = 0;
    meIntegrityGainSwitch[i] = 0;
    meIntegrityGainSwitchStay[i] = 0;
    meIntegrityTTId[i] = 0;
    meIntegrityTTBlockSize[i] = 0;
    meIntegrityMemChId[i] = 0;
    meIntegrityMemGain[i] = 0;
    meIntegrityMemTTId[i] = 0;
    meIntegrityMemTTBlockSize[i] = 0;
  }

}


EBIntegrityTask::~EBIntegrityTask(){

}

void EBIntegrityTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask");
    dbe_->rmdir("EcalBarrel/EBIntegrityTask");
  }

}

void EBIntegrityTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask");

    // checking when number of towers in data different than expected from header
    sprintf(histo, "EBIT DCC size error");
    meIntegrityDCCSize = dbe_->book1D(histo, histo, 36, 1, 37.);

    // checking when the gain is 0
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/Gain");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT gain %s", Numbers::sEB(i+1).c_str());
      meIntegrityGain[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe_->tag(meIntegrityGain[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/ChId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT ChId %s", Numbers::sEB(i+1).c_str());
      meIntegrityChId[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe_->tag(meIntegrityChId[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitch");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT gain switch %s", Numbers::sEB(i+1).c_str());
      meIntegrityGainSwitch[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe_->tag(meIntegrityGainSwitch[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitchStay");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT gain switch stay %s", Numbers::sEB(i+1).c_str());
      meIntegrityGainSwitchStay[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      dbe_->tag(meIntegrityGainSwitchStay[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT TTId %s", Numbers::sEB(i+1).c_str());
      meIntegrityTTId[i] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      dbe_->tag(meIntegrityTTId[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid size
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTBlockSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT TTBlockSize %s", Numbers::sEB(i+1).c_str());
      meIntegrityTTBlockSize[i] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      dbe_->tag(meIntegrityTTBlockSize[i], i+1);
    }

    // checking when mem channels have unexpected ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemChId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemChId %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemChId[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      dbe_->tag(meIntegrityMemChId[i], i+1);
    }

    // checking when mem samples have second bit encoding the gain different from 0
    // note: strictly speaking, this does not corrupt the mem sample gain value (since only first bit is considered)
    // but indicates that data are not completely correct
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemGain");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemGain %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemGain[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      dbe_->tag(meIntegrityMemGain[i], i+1);
    }

    // checking when mem tower block has unexpected ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemTTId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemTTId %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemTTId[i] = dbe_->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
      dbe_->tag(meIntegrityMemTTId[i], i+1);
    }

    // checking when mem tower block has invalid size
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemSize %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemTTBlockSize[i] = dbe_->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
      dbe_->tag(meIntegrityMemTTBlockSize[i], i+1);
    }

  }

}

void EBIntegrityTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask");

    if ( meIntegrityDCCSize ) dbe_->removeElement( meIntegrityDCCSize->getName() );
    meIntegrityDCCSize = 0;

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/Gain");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityGain[i] ) dbe_->removeElement( meIntegrityGain[i]->getName() );
      meIntegrityGain[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/ChId");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityChId[i] ) dbe_->removeElement( meIntegrityChId[i]->getName() );
      meIntegrityChId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitch");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityGainSwitch[i] ) dbe_->removeElement( meIntegrityGainSwitch[i]->getName() );
      meIntegrityGainSwitch[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitchStay");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityGainSwitchStay[i] ) dbe_->removeElement( meIntegrityGainSwitchStay[i]->getName() );
      meIntegrityGainSwitchStay[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTId");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityTTId[i] ) dbe_->removeElement( meIntegrityTTId[i]->getName() );
      meIntegrityTTId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTBlockSize");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityTTBlockSize[i] ) dbe_->removeElement( meIntegrityTTBlockSize[i]->getName() );
      meIntegrityTTBlockSize[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemChId");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityMemChId[i] ) dbe_->removeElement( meIntegrityMemChId[i]->getName() );
      meIntegrityMemChId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemGain");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityMemGain[i] ) dbe_->removeElement( meIntegrityMemGain[i]->getName() );
      meIntegrityMemGain[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemTTId");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityMemTTId[i] ) dbe_->removeElement( meIntegrityMemTTId[i]->getName() );
      meIntegrityMemTTId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemSize");
    for (int i = 0; i < 36 ; i++) {
      if ( meIntegrityMemTTBlockSize[i] ) dbe_->removeElement( meIntegrityMemTTBlockSize[i]->getName() );
      meIntegrityMemTTBlockSize[i] = 0;
    }

  }

  init_ = false;

}

void EBIntegrityTask::endJob(void){

  LogInfo("EBIntegrityTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBIntegrityTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<EBDetIdCollection> ids0;
    e.getByLabel(EBDetIdCollection0_, ids0);

    for ( EBDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++ idItr ) {

      EBDetId id = (*idItr);

      int ism = Numbers::iSM( id );

      float xism = ism - 0.5;

      if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EBDetIdCollection0_ << " not available";

  }

  try {

    Handle<EBDetIdCollection> ids1;
    e.getByLabel(EBDetIdCollection1_, ids1);

    for ( EBDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++ idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityGain[ism-1] ) meIntegrityGain[ism-1]->Fill(xie, xip);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EBDetIdCollection1_ << " not available";

  }

  try {

    Handle<EBDetIdCollection> ids2;
    e.getByLabel(EBDetIdCollection2_, ids2);

    for ( EBDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++ idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityChId[ism-1] ) meIntegrityChId[ism-1]->Fill(xie, xip);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EBDetIdCollection2_ << " not available";

  }

  try {

    Handle<EBDetIdCollection> ids3;
    e.getByLabel(EBDetIdCollection3_, ids3);

    for ( EBDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++ idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityGainSwitch[ism-1] ) meIntegrityGainSwitch[ism-1]->Fill(xie, xip);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EBDetIdCollection3_ << " not available";

  }

  try {

    Handle<EBDetIdCollection> ids4;
    e.getByLabel(EBDetIdCollection4_, ids4);

    for ( EBDetIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++ idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityGainSwitchStay[ism-1] ) meIntegrityGainSwitchStay[ism-1]->Fill(xie, xip);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EBDetIdCollection4_ << " not available";

  }

  try {

    Handle<EcalTrigTowerDetIdCollection> ids5;
    e.getByLabel(EcalTrigTowerDetIdCollection1_, ids5);

    for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++ idItr ) {

      EcalTrigTowerDetId id = (*idItr);

      int iet = id.ieta();
      int ipt = id.iphi();

      // phi_tower: change the range from global to SM-local
      ipt     = ( (ipt-1) % 4) +1;

      // phi_tower: range matters too
      //    if ( id.zside() >0)
      //      { ipt = 5 - ipt;      }

      int ismt = Numbers::iSM( id );

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      if ( meIntegrityTTId[ismt-1] ) meIntegrityTTId[ismt-1]->Fill(xiet, xipt);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EcalTrigTowerDetIdCollection1_ << " not available";

  }

  try {

    Handle<EcalTrigTowerDetIdCollection> ids6;
    e.getByLabel(EcalTrigTowerDetIdCollection2_, ids6);

    for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++ idItr ) {

      EcalTrigTowerDetId id = (*idItr);

      int iet = id.ieta();
      int ipt = id.iphi();


      // phi_tower: change the range from global to SM-local
      ipt     = ( (ipt-1) % 4) +1;

      // phi_tower: range matters too
      //    if ( id.zside() >0)
      //      { ipt = 5 - ipt;      }

      int ismt = Numbers::iSM( id );

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      if ( meIntegrityTTBlockSize[ismt-1] ) meIntegrityTTBlockSize[ismt-1]->Fill(xiet, xipt);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EcalTrigTowerDetIdCollection2_ << " not available";

  }

  try {

    Handle<EcalElectronicsIdCollection> ids7;
    e.getByLabel(EcalElectronicsIdCollection1_, ids7);

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      int ism = Numbers::iSM( id );

      int itt   = id.towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTId[ism-1] ) meIntegrityMemTTId[ism-1]->Fill(iTt,0);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  try {

    Handle<EcalElectronicsIdCollection> ids8;
    e.getByLabel(EcalElectronicsIdCollection2_, ids8);

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      int ism = Numbers::iSM( id );

      int itt   = id.towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTBlockSize[ism-1] ) meIntegrityMemTTBlockSize[ism-1]->Fill(iTt,0);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  try {

    Handle<EcalElectronicsIdCollection> ids9;
    e.getByLabel(EcalElectronicsIdCollection3_, ids9);

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      int ism = Numbers::iSM( id );

      int chid = id.channelId();
      int ie = EBIntegrityTask::chMemAbscissa[chid-1];
      int ip = EBIntegrityTask::chMemOrdinate[chid-1];

      int iTt = id.towerId();
      ie += (iTt-69)*5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityMemChId[ism-1] ) meIntegrityMemChId[ism-1]->Fill(xie,xip);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  try {

    Handle<EcalElectronicsIdCollection> ids10;
    e.getByLabel(EcalElectronicsIdCollection4_, ids10);

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids10->begin(); idItr != ids10->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      int ism = Numbers::iSM( id );

      int chid = id.channelId();
      int ie = EBIntegrityTask::chMemAbscissa[chid-1];
      int ip = EBIntegrityTask::chMemOrdinate[chid-1];

      int iTt = id.towerId();
      ie += (iTt-69)*5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityMemGain[ism-1] ) meIntegrityMemGain[ism-1]->Fill(xie,xip);

    }

  } catch ( exception& ex) {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection4_ << " not available";

  }

}//  end analyze

const int  EBIntegrityTask::chMemAbscissa [25] = {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5
};

const int  EBIntegrityTask::chMemOrdinate [25] = {
    1, 2, 3, 4, 5,
    5, 4, 3, 2, 1,
    1, 2, 3, 4, 5,
    5, 4, 3, 2, 1,
    1, 2, 3, 4, 5
};

