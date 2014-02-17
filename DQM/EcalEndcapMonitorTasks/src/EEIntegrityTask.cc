/*
 * \file EEIntegrityTask.cc
 *
 * $Date: 2012/04/27 13:46:15 $
 * $Revision: 1.61 $
 * \author G. Della Ricca
 *
 */

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEIntegrityTask.h"

EEIntegrityTask::EEIntegrityTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

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

  meIntegrityDCCSize = 0;
  for (int i = 0; i < 18; i++) {
    meIntegrityGain[i] = 0;
    meIntegrityChId[i] = 0;
    meIntegrityGainSwitch[i] = 0;
    meIntegrityTTId[i] = 0;
    meIntegrityTTBlockSize[i] = 0;
    meIntegrityMemChId[i] = 0;
    meIntegrityMemGain[i] = 0;
    meIntegrityMemTTId[i] = 0;
    meIntegrityMemTTBlockSize[i] = 0;
  }
  meIntegrityErrorsByLumi = 0;
}


EEIntegrityTask::~EEIntegrityTask(){

}

void EEIntegrityTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEIntegrityTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EEIntegrityTask/" + subfolder_);
    dqmStore_->rmdir(prefixME_ + "/EEIntegrityTask");
  }

}

void EEIntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup) {

  if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Reset();

}

void EEIntegrityTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {
}

void EEIntegrityTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EEIntegrityTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EEIntegrityTask::reset(void) {

  if ( meIntegrityDCCSize ) meIntegrityDCCSize->Reset();
  for (int i = 0; i < 18; i++) {
    if ( meIntegrityGain[i] ) meIntegrityGain[i]->Reset();
    if ( meIntegrityChId[i] ) meIntegrityChId[i]->Reset();
    if ( meIntegrityGainSwitch[i] ) meIntegrityGainSwitch[i]->Reset();
    if ( meIntegrityTTId[i] ) meIntegrityTTId[i]->Reset();
    if ( meIntegrityTTBlockSize[i] ) meIntegrityTTBlockSize[i]->Reset();
    if ( meIntegrityMemChId[i] ) meIntegrityMemChId[i]->Reset();
    if ( meIntegrityMemGain[i] ) meIntegrityMemGain[i]->Reset();
    if ( meIntegrityMemTTId[i] ) meIntegrityMemTTId[i]->Reset();
    if ( meIntegrityMemTTBlockSize[i] ) meIntegrityMemTTBlockSize[i]->Reset();
  }
  if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Reset();

}

void EEIntegrityTask::setup(void){

  init_ = true;

  std::string name;
  std::string dir;

  if ( dqmStore_ ) {

    dir = prefixME_ + "/EEIntegrityTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir);

    // checking when number of towers in data different than expected from header
    name = "EEIT DCC size error";
    meIntegrityDCCSize = dqmStore_->book1D(name, name, 18, 1., 19.);
    for (int i = 0; i < 18; i++) {
      meIntegrityDCCSize->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    // checking the number of integrity errors in each DCC for each lumi
    // crystal integrity error is weighted by 1/850
    // tower integrity error is weighted by 1/34
    // bin 0 contains the number of processed events in the lumi (for normalization)
    name = "EEIT weighted integrity errors by lumi";
    meIntegrityErrorsByLumi = dqmStore_->book1D(name, name, 18, 1., 19.);
    meIntegrityErrorsByLumi->setLumiFlag();
    for (int i = 0; i < 18; i++) {
      meIntegrityErrorsByLumi->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    // checking when the gain is 0
    dqmStore_->setCurrentFolder(dir + "/Gain");
    for (int i = 0; i < 18; i++) {
      name = "EEIT gain " + Numbers::sEE(i+1);
      meIntegrityGain[i] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityGain[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meIntegrityGain[i]->setAxisTitle("101-ix", 1);
      meIntegrityGain[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meIntegrityGain[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dqmStore_->setCurrentFolder(dir + "/ChId");
    for (int i = 0; i < 18; i++) {
      name = "EEIT ChId " + Numbers::sEE(i+1);
      meIntegrityChId[i] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityChId[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meIntegrityChId[i]->setAxisTitle("101-ix", 1);
      meIntegrityChId[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meIntegrityChId[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dqmStore_->setCurrentFolder(dir + "/GainSwitch");
    for (int i = 0; i < 18; i++) {
      name = "EEIT gain switch " + Numbers::sEE(i+1);
      meIntegrityGainSwitch[i] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityGainSwitch[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meIntegrityGainSwitch[i]->setAxisTitle("101-ix", 1);
      meIntegrityGainSwitch[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meIntegrityGainSwitch[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid ID
    dqmStore_->setCurrentFolder(dir + "/TTId");
    for (int i = 0; i < 18; i++) {
      name = "EEIT TTId " + Numbers::sEE(i+1);
      meIntegrityTTId[i] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityTTId[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meIntegrityTTId[i]->setAxisTitle("101-ix", 1);
      meIntegrityTTId[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meIntegrityTTId[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid size
    dqmStore_->setCurrentFolder(dir + "/TTBlockSize");
    for (int i = 0; i < 18; i++) {
      name = "EEIT TTBlockSize " + Numbers::sEE(i+1);
      meIntegrityTTBlockSize[i] = dqmStore_->book2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityTTBlockSize[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meIntegrityTTBlockSize[i]->setAxisTitle("101-ix", 1);
      meIntegrityTTBlockSize[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meIntegrityTTBlockSize[i], i+1);
    }

    // checking when mem channels have unexpected ID
    dqmStore_->setCurrentFolder(dir + "/MemChId");
    for (int i = 0; i < 18; i++) {
      name = "EEIT MemChId " + Numbers::sEE(i+1);
      meIntegrityMemChId[i] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemChId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemChId[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemChId[i], i+1);
    }

    // checking when mem samples have second bit encoding the gain different from 0
    // note: strictly speaking, this does not corrupt the mem sample gain value (since only first bit is considered)
    // but indicates that data are not completely correct
    dqmStore_->setCurrentFolder(dir + "/MemGain");
    for (int i = 0; i < 18; i++) {
      name = "EEIT MemGain " + Numbers::sEE(i+1);
      meIntegrityMemGain[i] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemGain[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemGain[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemGain[i], i+1);
    }

    // checking when mem tower block has unexpected ID
    dqmStore_->setCurrentFolder(dir + "/MemTTId");
    for (int i = 0; i < 18; i++) {
      name = "EEIT MemTTId " + Numbers::sEE(i+1);
      meIntegrityMemTTId[i] = dqmStore_->book2D(name, name, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemTTId[i], i+1);
    }

    // checking when mem tower block has invalid size
    dqmStore_->setCurrentFolder(dir + "/MemSize");
    for (int i = 0; i < 18; i++) {
      name = "EEIT MemSize " + Numbers::sEE(i+1);
      meIntegrityMemTTBlockSize[i] = dqmStore_->book2D(name, name, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTBlockSize[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTBlockSize[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemTTBlockSize[i], i+1);
    }

  }

}

void EEIntegrityTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    std::string dir;

    dir = prefixME_ + "/EEIntegrityTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir + "");

    if ( meIntegrityDCCSize ) dqmStore_->removeElement( meIntegrityDCCSize->getName() );
    meIntegrityDCCSize = 0;

    if ( meIntegrityErrorsByLumi ) dqmStore_->removeElement( meIntegrityErrorsByLumi->getName() );
    meIntegrityErrorsByLumi = 0;

    dqmStore_->setCurrentFolder(dir + "/Gain");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityGain[i] ) dqmStore_->removeElement( meIntegrityGain[i]->getName() );
      meIntegrityGain[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/ChId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityChId[i] ) dqmStore_->removeElement( meIntegrityChId[i]->getName() );
      meIntegrityChId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/GainSwitch");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityGainSwitch[i] ) dqmStore_->removeElement( meIntegrityGainSwitch[i]->getName() );
      meIntegrityGainSwitch[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/TTId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityTTId[i] ) dqmStore_->removeElement( meIntegrityTTId[i]->getName() );
      meIntegrityTTId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/TTBlockSize");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityTTBlockSize[i] ) dqmStore_->removeElement( meIntegrityTTBlockSize[i]->getName() );
      meIntegrityTTBlockSize[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemChId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemChId[i] ) dqmStore_->removeElement( meIntegrityMemChId[i]->getName() );
      meIntegrityMemChId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemGain");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemGain[i] ) dqmStore_->removeElement( meIntegrityMemGain[i]->getName() );
      meIntegrityMemGain[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemTTId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemTTId[i] ) dqmStore_->removeElement( meIntegrityMemTTId[i]->getName() );
      meIntegrityMemTTId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemSize");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemTTBlockSize[i] ) dqmStore_->removeElement( meIntegrityMemTTBlockSize[i]->getName() );
      meIntegrityMemTTBlockSize[i] = 0;
    }

  }

  init_ = false;

}

void EEIntegrityTask::endJob(void){

  edm::LogInfo("EEIntegrityTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // fill bin 0 with number of events in the lumi
  if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(0.);

  edm::Handle<EEDetIdCollection> ids0;

  if ( e.getByLabel(EEDetIdCollection0_, ids0) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++idItr ) {

      int ism = Numbers::iSM( *idItr );

      float xism = ism + 0.5;

      if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

    }

  } else {

//    edm::LogWarning("EEIntegrityTask") << EEDetIdCollection0_ << " not available";

  }

  edm::Handle<EEDetIdCollection> ids1;

  if ( e.getByLabel(EEDetIdCollection1_, ids1) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityGain[ism-1] ) meIntegrityGain[ism-1]->Fill(xix, xiy);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./850.);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EEDetIdCollection1_ << " not available";

  }

  edm::Handle<EEDetIdCollection> ids2;

  if ( e.getByLabel(EEDetIdCollection2_, ids2) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityChId[ism-1] ) meIntegrityChId[ism-1]->Fill(xix, xiy);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./850.);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EEDetIdCollection2_ << " not available";

  }

  edm::Handle<EEDetIdCollection> ids3;

  if ( e.getByLabel(EEDetIdCollection3_, ids3) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityGainSwitch[ism-1] ) meIntegrityGainSwitch[ism-1]->Fill(xix, xiy);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./850.);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EEDetIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );
      float xism = ism + 0.5;

      std::vector<DetId>* crystals = Numbers::crystals( *idItr );

      for ( unsigned int i=0; i<crystals->size(); i++ ) {

      EEDetId id = (*crystals)[i];

      int ix = id.ix();
      int iy = id.iy();

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityTTId[ism-1] ) meIntegrityTTId[ism-1]->Fill(xix, xiy);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./34./crystals->size());

      }

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );
      float xism = ism + 0.5;

      std::vector<DetId>* crystals = Numbers::crystals( *idItr );

      for ( unsigned int i=0; i<crystals->size(); i++ ) {

      EEDetId id = (*crystals)[i];

      int ix = id.ix();
      int iy = id.iy();

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityTTBlockSize[ism-1] ) meIntegrityTTBlockSize[ism-1]->Fill(xix, xiy);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./34./crystals->size());

      }

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );

      int itt   = idItr->towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTId[ism-1] ) meIntegrityMemTTId[ism-1]->Fill(iTt,0);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );

      int itt   = idItr->towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTBlockSize[ism-1] ) meIntegrityMemTTBlockSize[ism-1]->Fill(iTt,0);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids8;

  if (  e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );

      int chid = idItr->channelId();
      int ie = EEIntegrityTask::chMemAbscissa[chid-1];
      int ip = EEIntegrityTask::chMemOrdinate[chid-1];

      int itt = idItr->towerId();
      ie += (itt-69)*5;

      float xix = ie - 0.5;
      float xiy = ip - 0.5;

      if ( meIntegrityMemChId[ism-1] ) meIntegrityMemChId[ism-1]->Fill(xix,xiy);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );

      int chid = idItr->channelId();
      int ie = EEIntegrityTask::chMemAbscissa[chid-1];
      int ip = EEIntegrityTask::chMemOrdinate[chid-1];

      int itt = idItr->towerId();
      ie += (itt-69)*5;

      float xix = ie - 0.5;
      float xiy = ip - 0.5;

      if ( meIntegrityMemGain[ism-1] ) meIntegrityMemGain[ism-1]->Fill(xix,xiy);

    }

  } else {

    edm::LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection6_ << " not available";

  }

}//  end analyze

const int  EEIntegrityTask::chMemAbscissa [25] = {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5
};

const int  EEIntegrityTask::chMemOrdinate [25] = {
    1, 2, 3, 4, 5,
    5, 4, 3, 2, 1,
    1, 2, 3, 4, 5,
    5, 4, 3, 2, 1,
    1, 2, 3, 4, 5
};

