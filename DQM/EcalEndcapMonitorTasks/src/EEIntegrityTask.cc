/*
 * \file EEIntegrityTask.cc
 *
 * $Date: 2008/03/14 14:57:58 $
 * $Revision: 1.32 $
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

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEIntegrityTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEIntegrityTask::EEIntegrityTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

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

}


EEIntegrityTask::~EEIntegrityTask(){

}

void EEIntegrityTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask");
    dbe_->rmdir("EcalEndcap/EEIntegrityTask");
  }

  Numbers::initGeometry(c);

}

void EEIntegrityTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask");

    // checking when number of towers in data different than expected from header
    sprintf(histo, "EEIT DCC size error");
    meIntegrityDCCSize = dbe_->book1D(histo, histo, 18, 1, 19.);
    for (int i = 0; i < 18; i++) {
      meIntegrityDCCSize->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    // checking when the gain is 0
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/Gain");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT gain %s", Numbers::sEE(i+1).c_str());
      meIntegrityGain[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityGain[i]->setAxisTitle("jx", 1);
      meIntegrityGain[i]->setAxisTitle("jy", 2);
      dbe_->tag(meIntegrityGain[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/ChId");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT ChId %s", Numbers::sEE(i+1).c_str());
      meIntegrityChId[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityChId[i]->setAxisTitle("jx", 1);
      meIntegrityChId[i]->setAxisTitle("jy", 2);
      dbe_->tag(meIntegrityChId[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/GainSwitch");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT gain switch %s", Numbers::sEE(i+1).c_str());
      meIntegrityGainSwitch[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityGainSwitch[i]->setAxisTitle("jx", 1);
      meIntegrityGainSwitch[i]->setAxisTitle("jy", 2);
      dbe_->tag(meIntegrityGainSwitch[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/TTId");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT TTId %s", Numbers::sEE(i+1).c_str());
      meIntegrityTTId[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityTTId[i]->setAxisTitle("jx", 1);
      meIntegrityTTId[i]->setAxisTitle("jy", 2);
      dbe_->tag(meIntegrityTTId[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid size
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/TTBlockSize");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT TTBlockSize %s", Numbers::sEE(i+1).c_str());
      meIntegrityTTBlockSize[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meIntegrityTTBlockSize[i]->setAxisTitle("jx", 1);
      meIntegrityTTBlockSize[i]->setAxisTitle("jy", 2);
      dbe_->tag(meIntegrityTTBlockSize[i], i+1);
    }

    // checking when mem channels have unexpected ID
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemChId");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT MemChId %s", Numbers::sEE(i+1).c_str());
      meIntegrityMemChId[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemChId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemChId[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemChId[i], i+1);
    }

    // checking when mem samples have second bit encoding the gain different from 0
    // note: strictly speaking, this does not corrupt the mem sample gain value (since only first bit is considered)
    // but indicates that data are not completely correct
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemGain");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT MemGain %s", Numbers::sEE(i+1).c_str());
      meIntegrityMemGain[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemGain[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemGain[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemGain[i], i+1);
    }

    // checking when mem tower block has unexpected ID
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemTTId");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT MemTTId %s", Numbers::sEE(i+1).c_str());
      meIntegrityMemTTId[i] = dbe_->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemTTId[i], i+1);
    }

    // checking when mem tower block has invalid size
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemSize");
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEIT MemSize %s", Numbers::sEE(i+1).c_str());
      meIntegrityMemTTBlockSize[i] = dbe_->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemTTBlockSize[i], i+1);
    }

  }

}

void EEIntegrityTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask");

    if ( meIntegrityDCCSize ) dbe_->removeElement( meIntegrityDCCSize->getName() );
    meIntegrityDCCSize = 0;

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/Gain");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityGain[i] ) dbe_->removeElement( meIntegrityGain[i]->getName() );
      meIntegrityGain[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/ChId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityChId[i] ) dbe_->removeElement( meIntegrityChId[i]->getName() );
      meIntegrityChId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/GainSwitch");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityGainSwitch[i] ) dbe_->removeElement( meIntegrityGainSwitch[i]->getName() );
      meIntegrityGainSwitch[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/TTId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityTTId[i] ) dbe_->removeElement( meIntegrityTTId[i]->getName() );
      meIntegrityTTId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/TTBlockSize");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityTTBlockSize[i] ) dbe_->removeElement( meIntegrityTTBlockSize[i]->getName() );
      meIntegrityTTBlockSize[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemChId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemChId[i] ) dbe_->removeElement( meIntegrityMemChId[i]->getName() );
      meIntegrityMemChId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemGain");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemGain[i] ) dbe_->removeElement( meIntegrityMemGain[i]->getName() );
      meIntegrityMemGain[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemTTId");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemTTId[i] ) dbe_->removeElement( meIntegrityMemTTId[i]->getName() );
      meIntegrityMemTTId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalEndcap/EEIntegrityTask/MemSize");
    for (int i = 0; i < 18; i++) {
      if ( meIntegrityMemTTBlockSize[i] ) dbe_->removeElement( meIntegrityMemTTBlockSize[i]->getName() );
      meIntegrityMemTTBlockSize[i] = 0;
    }

  }

  init_ = false;

}

void EEIntegrityTask::endJob(void){

  LogInfo("EEIntegrityTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEIntegrityTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EEDetIdCollection> ids0;

  if ( e.getByLabel(EEDetIdCollection0_, ids0) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ism = Numbers::iSM( id );

      float xism = ism - 0.5;

      if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

    }

  } else {

    LogWarning("EEIntegrityTask") << EEDetIdCollection0_ << " not available";

  }

  Handle<EEDetIdCollection> ids1;

  if ( e.getByLabel(EEDetIdCollection1_, ids1) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityGain[ism-1] ) meIntegrityGain[ism-1]->Fill(xix, xiy);

    }

  } else {

    LogWarning("EEIntegrityTask") << EEDetIdCollection1_ << " not available";

  }

  Handle<EEDetIdCollection> ids2;

  if ( e.getByLabel(EEDetIdCollection2_, ids2) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityChId[ism-1] ) meIntegrityChId[ism-1]->Fill(xix, xiy);

    }

  } else {

    LogWarning("EEIntegrityTask") << EEDetIdCollection2_ << " not available";

  }

  Handle<EEDetIdCollection> ids3;

  if ( e.getByLabel(EEDetIdCollection3_, ids3) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++idItr ) {

      EEDetId id = (*idItr);

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityGainSwitch[ism-1] ) meIntegrityGainSwitch[ism-1]->Fill(xix, xiy);

    }

  } else {

    LogWarning("EEIntegrityTask") << EEDetIdCollection3_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      vector<DetId> crystals = Numbers::crystals( id );

      for ( unsigned int i=0; i<crystals.size(); i++ ) {

      EEDetId id = crystals[i];

      int ix = id.ix();
      int iy = id.iy();

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityTTId[ism-1] ) meIntegrityTTId[ism-1]->Fill(xix, xiy);

      }

    }

  } else {

    LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      vector<DetId> crystals = Numbers::crystals( id );

      for ( unsigned int i=0; i<crystals.size(); i++ ) {

      EEDetId id = crystals[i];

      int ix = id.ix();
      int iy = id.iy();

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      if ( meIntegrityTTBlockSize[ism-1] ) meIntegrityTTBlockSize[ism-1]->Fill(xix, xiy);

      }

    }

  } else {

    LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      int itt   = id.towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTId[ism-1] ) meIntegrityMemTTId[ism-1]->Fill(iTt,0);

    }

  } else {

    LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      int itt   = id.towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTBlockSize[ism-1] ) meIntegrityMemTTBlockSize[ism-1]->Fill(iTt,0);

    }

  } else {

    LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids8;

  if (  e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      int chid = id.channelId();
      int ie = EEIntegrityTask::chMemAbscissa[chid-1];
      int ip = EEIntegrityTask::chMemOrdinate[chid-1];

      int itt = id.towerId();
      ie += (itt-69)*5;

      float xix = ie - 0.5;
      float xiy = ip - 0.5;

      if ( meIntegrityMemChId[ism-1] ) meIntegrityMemChId[ism-1]->Fill(xix,xiy);

    }

  } else {

    LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( id );

      int chid = id.channelId();
      int ie = EEIntegrityTask::chMemAbscissa[chid-1];
      int ip = EEIntegrityTask::chMemOrdinate[chid-1];

      int itt = id.towerId();
      ie += (itt-69)*5;

      float xix = ie - 0.5;
      float xiy = ip - 0.5;

      if ( meIntegrityMemGain[ism-1] ) meIntegrityMemGain[ism-1]->Fill(xix,xiy);

    }

  } else {

    LogWarning("EEIntegrityTask") << EcalElectronicsIdCollection6_ << " not available";

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

