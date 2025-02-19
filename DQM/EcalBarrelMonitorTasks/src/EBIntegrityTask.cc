/*
 * \file EBIntegrityTask.cc
 *
 * $Date: 2012/04/27 13:46:02 $
 * $Revision: 1.89 $
 * \author G. Della Ricca
 *
 */

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h"

EBIntegrityTask::EBIntegrityTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  subfolder_ = ps.getUntrackedParameter<std::string>("subfolder", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

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

  meIntegrityDCCSize = 0;
  for (int i = 0; i < 36; i++) {
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


EBIntegrityTask::~EBIntegrityTask(){

}

void EBIntegrityTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBIntegrityTask");
    if(subfolder_.size())
      dqmStore_->setCurrentFolder(prefixME_ + "/EBIntegrityTask/" + subfolder_);
    dqmStore_->rmdir(prefixME_ + "/EBIntegrityTask");
  }

}

void EBIntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup) {

  if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Reset();

}

void EBIntegrityTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {
}

void EBIntegrityTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBIntegrityTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBIntegrityTask::reset(void) {

  if ( meIntegrityDCCSize ) meIntegrityDCCSize->Reset();
  for (int i = 0; i < 36; i++) {
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

void EBIntegrityTask::setup(void){

  init_ = true;

  std::string name;
  std::string dir;

  if ( dqmStore_ ) {

    dir = prefixME_ + "/EBIntegrityTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir);

    // checking when number of towers in data different than expected from header
    name = "EBIT DCC size error";
    meIntegrityDCCSize = dqmStore_->book1D(name, name, 36, 1., 37.);
    for (int i = 0; i < 36; i++) {
      meIntegrityDCCSize->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    // checking the number of integrity errors in each DCC for each lumi
    // crystal integrity error is weighted by 1/1700
    // tower integrity error is weighted by 1/68
    // bin 0 contains the number of processed events in the lumi (for normalization)
    name = "EBIT weighted integrity errors by lumi";
    meIntegrityErrorsByLumi = dqmStore_->book1D(name, name, 36, 1., 37.);
    meIntegrityErrorsByLumi->setLumiFlag();
    for (int i = 0; i < 36; i++) {
      meIntegrityErrorsByLumi->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    // checking when the gain is 0
    dqmStore_->setCurrentFolder(dir + "/Gain");
    for (int i = 0; i < 36; i++) {
      name = "EBIT gain " + Numbers::sEB(i+1);
      meIntegrityGain[i] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meIntegrityGain[i]->setAxisTitle("ieta", 1);
      meIntegrityGain[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(meIntegrityGain[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dqmStore_->setCurrentFolder(dir + "/ChId");
    for (int i = 0; i < 36; i++) {
      name = "EBIT ChId " + Numbers::sEB(i+1);
      meIntegrityChId[i] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meIntegrityChId[i]->setAxisTitle("ieta", 1);
      meIntegrityChId[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(meIntegrityChId[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dqmStore_->setCurrentFolder(dir + "/GainSwitch");
    for (int i = 0; i < 36; i++) {
      name = "EBIT gain switch " + Numbers::sEB(i+1);
      meIntegrityGainSwitch[i] = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);
      meIntegrityGainSwitch[i]->setAxisTitle("ieta", 1);
      meIntegrityGainSwitch[i]->setAxisTitle("iphi", 2);
      dqmStore_->tag(meIntegrityGainSwitch[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid ID
    dqmStore_->setCurrentFolder(dir + "/TTId");
    for (int i = 0; i < 36; i++) {
      name = "EBIT TTId " + Numbers::sEB(i+1);
      meIntegrityTTId[i] = dqmStore_->book2D(name, name, 17, 0., 17., 4, 0., 4.);
      meIntegrityTTId[i]->setAxisTitle("ieta'", 1);
      meIntegrityTTId[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meIntegrityTTId[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid size
    dqmStore_->setCurrentFolder(dir + "/TTBlockSize");
    for (int i = 0; i < 36; i++) {
      name = "EBIT TTBlockSize " + Numbers::sEB(i+1);
      meIntegrityTTBlockSize[i] = dqmStore_->book2D(name, name, 17, 0., 17., 4, 0., 4.);
      meIntegrityTTBlockSize[i]->setAxisTitle("ieta'", 1);
      meIntegrityTTBlockSize[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meIntegrityTTBlockSize[i], i+1);
    }

    // checking when mem channels have unexpected ID
    dqmStore_->setCurrentFolder(dir + "/MemChId");
    for (int i = 0; i < 36; i++) {
      name = "EBIT MemChId " + Numbers::sEB(i+1);
      meIntegrityMemChId[i] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemChId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemChId[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemChId[i], i+1);
    }

    // checking when mem samples have second bit encoding the gain different from 0
    // note: strictly speaking, this does not corrupt the mem sample gain value (since only first bit is considered)
    // but indicates that data are not completely correct
    dqmStore_->setCurrentFolder(dir + "/MemGain");
    for (int i = 0; i < 36; i++) {
      name = "EBIT MemGain " + Numbers::sEB(i+1);
      meIntegrityMemGain[i] = dqmStore_->book2D(name, name, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemGain[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemGain[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemGain[i], i+1);
    }

    // checking when mem tower block has unexpected ID
    dqmStore_->setCurrentFolder(dir + "/MemTTId");
    for (int i = 0; i < 36; i++) {
      name = "EBIT MemTTId " + Numbers::sEB(i+1);
      meIntegrityMemTTId[i] = dqmStore_->book2D(name, name, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemTTId[i], i+1);
    }

    // checking when mem tower block has invalid size
    dqmStore_->setCurrentFolder(dir + "/MemSize");
    for (int i = 0; i < 36; i++) {
      name = "EBIT MemSize " + Numbers::sEB(i+1);
      meIntegrityMemTTBlockSize[i] = dqmStore_->book2D(name, name, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTBlockSize[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTBlockSize[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTBlockSize[i]->setAxisTitle("channel", 2);
      dqmStore_->tag(meIntegrityMemTTBlockSize[i], i+1);
    }

  }

}

void EBIntegrityTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    std::string dir;

    dir = prefixME_ + "/EBIntegrityTask";
    if(subfolder_.size())
      dir += "/" + subfolder_;

    dqmStore_->setCurrentFolder(dir + "");

    if ( meIntegrityDCCSize ) dqmStore_->removeElement( meIntegrityDCCSize->getName() );
    meIntegrityDCCSize = 0;

    if ( meIntegrityErrorsByLumi ) dqmStore_->removeElement( meIntegrityErrorsByLumi->getName() );
    meIntegrityErrorsByLumi = 0;

    dqmStore_->setCurrentFolder(dir + "/Gain");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityGain[i] ) dqmStore_->removeElement( meIntegrityGain[i]->getName() );
      meIntegrityGain[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/ChId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityChId[i] ) dqmStore_->removeElement( meIntegrityChId[i]->getName() );
      meIntegrityChId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/GainSwitch");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityGainSwitch[i] ) dqmStore_->removeElement( meIntegrityGainSwitch[i]->getName() );
      meIntegrityGainSwitch[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/TTId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityTTId[i] ) dqmStore_->removeElement( meIntegrityTTId[i]->getName() );
      meIntegrityTTId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/TTBlockSize");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityTTBlockSize[i] ) dqmStore_->removeElement( meIntegrityTTBlockSize[i]->getName() );
      meIntegrityTTBlockSize[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemChId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemChId[i] ) dqmStore_->removeElement( meIntegrityMemChId[i]->getName() );
      meIntegrityMemChId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemGain");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemGain[i] ) dqmStore_->removeElement( meIntegrityMemGain[i]->getName() );
      meIntegrityMemGain[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemTTId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemTTId[i] ) dqmStore_->removeElement( meIntegrityMemTTId[i]->getName() );
      meIntegrityMemTTId[i] = 0;
    }

    dqmStore_->setCurrentFolder(dir + "/MemSize");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemTTBlockSize[i] ) dqmStore_->removeElement( meIntegrityMemTTBlockSize[i]->getName() );
      meIntegrityMemTTBlockSize[i] = 0;
    }

  }

  init_ = false;

}

void EBIntegrityTask::endJob(void){

  edm::LogInfo("EBIntegrityTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // fill bin 0 with number of events in the lumi
  if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(0.);

  edm::Handle<EBDetIdCollection> ids0;

  if ( e.getByLabel(EBDetIdCollection0_, ids0) ) {

    for ( EBDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++idItr ) {

      int ism = Numbers::iSM( *idItr );

      float xism = ism + 0.5;

      if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

    }

  } else {

//    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection0_ << " not available";

  }

  edm::Handle<EBDetIdCollection> ids1;

  if ( e.getByLabel(EBDetIdCollection1_, ids1) ) {

    for ( EBDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityGain[ism-1] ) meIntegrityGain[ism-1]->Fill(xie, xip);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./1700.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection1_ << " not available";

  }

  edm::Handle<EBDetIdCollection> ids2;

  if ( e.getByLabel(EBDetIdCollection2_, ids2) ) {

    for ( EBDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityChId[ism-1] ) meIntegrityChId[ism-1]->Fill(xie, xip);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./1700.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection2_ << " not available";

  }

  edm::Handle<EBDetIdCollection> ids3;

  if ( e.getByLabel(EBDetIdCollection3_, ids3) ) {

    for ( EBDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++idItr ) {

      EBDetId id = (*idItr);

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityGainSwitch[ism-1] ) meIntegrityGainSwitch[ism-1]->Fill(xie, xip);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism, 1./1700.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int itt = idItr->towerId();

      int iet = (itt-1)/4 + 1;
      int ipt = (itt-1)%4 + 1;

      int ismt = Numbers::iSM( *idItr );
      float xismt = ismt + 0.5;

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      if ( meIntegrityTTId[ismt-1] ) meIntegrityTTId[ismt-1]->Fill(xiet, xipt);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xismt, 1./68.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int itt = idItr->towerId();

      int iet = (itt-1)/4 + 1;
      int ipt = (itt-1)%4 + 1;

      int ismt = Numbers::iSM( *idItr );
      float xismt = ismt + 0.5;

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      if ( meIntegrityTTBlockSize[ismt-1] ) meIntegrityTTBlockSize[ismt-1]->Fill(xiet, xipt);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xismt, 1./68.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *idItr );

      int itt   = idItr->towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTId[ism-1] ) meIntegrityMemTTId[ism-1]->Fill(iTt,0);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *idItr );

      int itt   = idItr->towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTBlockSize[ism-1] ) meIntegrityMemTTBlockSize[ism-1]->Fill(iTt,0);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids8;

  if ( e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *idItr );

      int chid = idItr->channelId();
      int ie = EBIntegrityTask::chMemAbscissa[chid-1];
      int ip = EBIntegrityTask::chMemOrdinate[chid-1];

      int itt = idItr->towerId();
      ie += (itt-69)*5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityMemChId[ism-1] ) meIntegrityMemChId[ism-1]->Fill(xie,xip);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *idItr );

      int chid = idItr->channelId();
      int ie = EBIntegrityTask::chMemAbscissa[chid-1];
      int ip = EBIntegrityTask::chMemOrdinate[chid-1];

      int itt = idItr->towerId();
      ie += (itt-69)*5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityMemGain[ism-1] ) meIntegrityMemGain[ism-1]->Fill(xie,xip);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection6_ << " not available";

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

