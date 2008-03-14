/*
 * \file EBIntegrityTask.cc
 *
 * $Date: 2008/01/26 20:13:54 $
 * $Revision: 1.61 $
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

#include "DataFormats/EcalDetId/interface/EBDetId.h"

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

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

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

}


EBIntegrityTask::~EBIntegrityTask(){

}

void EBIntegrityTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask");
    dbe_->rmdir("EcalBarrel/EBIntegrityTask");
  }

  Numbers::initGeometry(c);

}

void EBIntegrityTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask");

    // checking when number of towers in data different than expected from header
    sprintf(histo, "EBIT DCC size error");
    meIntegrityDCCSize = dbe_->book1D(histo, histo, 36, 1, 37.);
    meIntegrityDCCSize->setAxisTitle("DCC module", 1);

    // checking when the gain is 0
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/Gain");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT gain %s", Numbers::sEB(i+1).c_str());
      meIntegrityGain[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meIntegrityGain[i]->setAxisTitle("ieta", 1);
      meIntegrityGain[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meIntegrityGain[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/ChId");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT ChId %s", Numbers::sEB(i+1).c_str());
      meIntegrityChId[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meIntegrityChId[i]->setAxisTitle("ieta", 1);
      meIntegrityChId[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meIntegrityChId[i], i+1);
    }

    // checking when channel has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitch");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT gain switch %s", Numbers::sEB(i+1).c_str());
      meIntegrityGainSwitch[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meIntegrityGainSwitch[i]->setAxisTitle("ieta", 1);
      meIntegrityGainSwitch[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meIntegrityGainSwitch[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTId");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT TTId %s", Numbers::sEB(i+1).c_str());
      meIntegrityTTId[i] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      meIntegrityTTId[i]->setAxisTitle("ieta'", 1);
      meIntegrityTTId[i]->setAxisTitle("iphi'", 2);
      dbe_->tag(meIntegrityTTId[i], i+1);
    }

    // checking when trigger tower has unexpected or invalid size
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTBlockSize");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT TTBlockSize %s", Numbers::sEB(i+1).c_str());
      meIntegrityTTBlockSize[i] = dbe_->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
      meIntegrityTTBlockSize[i]->setAxisTitle("ieta'", 1);
      meIntegrityTTBlockSize[i]->setAxisTitle("iphi'", 2);
      dbe_->tag(meIntegrityTTBlockSize[i], i+1);
    }

    // checking when mem channels have unexpected ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemChId");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT MemChId %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemChId[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemChId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemChId[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemChId[i], i+1);
    }

    // checking when mem samples have second bit encoding the gain different from 0
    // note: strictly speaking, this does not corrupt the mem sample gain value (since only first bit is considered)
    // but indicates that data are not completely correct
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemGain");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT MemGain %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemGain[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      meIntegrityMemGain[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemGain[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemGain[i], i+1);
    }

    // checking when mem tower block has unexpected ID
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemTTId");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT MemTTId %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemTTId[i] = dbe_->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("channel", 2);
      dbe_->tag(meIntegrityMemTTId[i], i+1);
    }

    // checking when mem tower block has invalid size
    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemSize");
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBIT MemSize %s", Numbers::sEB(i+1).c_str());
      meIntegrityMemTTBlockSize[i] = dbe_->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
      meIntegrityMemTTBlockSize[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("pseudo-strip", 1);
      meIntegrityMemTTId[i]->setAxisTitle("channel", 2);
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
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityGain[i] ) dbe_->removeElement( meIntegrityGain[i]->getName() );
      meIntegrityGain[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/ChId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityChId[i] ) dbe_->removeElement( meIntegrityChId[i]->getName() );
      meIntegrityChId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitch");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityGainSwitch[i] ) dbe_->removeElement( meIntegrityGainSwitch[i]->getName() );
      meIntegrityGainSwitch[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityTTId[i] ) dbe_->removeElement( meIntegrityTTId[i]->getName() );
      meIntegrityTTId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTBlockSize");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityTTBlockSize[i] ) dbe_->removeElement( meIntegrityTTBlockSize[i]->getName() );
      meIntegrityTTBlockSize[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemChId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemChId[i] ) dbe_->removeElement( meIntegrityMemChId[i]->getName() );
      meIntegrityMemChId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemGain");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemGain[i] ) dbe_->removeElement( meIntegrityMemGain[i]->getName() );
      meIntegrityMemGain[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemTTId");
    for (int i = 0; i < 36; i++) {
      if ( meIntegrityMemTTId[i] ) dbe_->removeElement( meIntegrityMemTTId[i]->getName() );
      meIntegrityMemTTId[i] = 0;
    }

    dbe_->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemSize");
    for (int i = 0; i < 36; i++) {
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

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDetIdCollection> ids0;

  if ( e.getByLabel(EBDetIdCollection0_, ids0) ) {

    for ( EBDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++ idItr ) {

      EBDetId id = (*idItr);

      int ism = Numbers::iSM( id );

      float xism = ism - 0.5;

      if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

    }

  } else {

    LogWarning("EBIntegrityTask") << EBDetIdCollection0_ << " not available";

  }

  Handle<EBDetIdCollection> ids1;

  if ( e.getByLabel(EBDetIdCollection1_, ids1) ) {

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

  } else {

    LogWarning("EBIntegrityTask") << EBDetIdCollection1_ << " not available";

  }

  Handle<EBDetIdCollection> ids2;

  if ( e.getByLabel(EBDetIdCollection2_, ids2) ) {

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

  } else {

    LogWarning("EBIntegrityTask") << EBDetIdCollection2_ << " not available";

  }

  Handle<EBDetIdCollection> ids3;

  if ( e.getByLabel(EBDetIdCollection3_, ids3) ) {

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

  } else {

    LogWarning("EBIntegrityTask") << EBDetIdCollection3_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int itt = id.towerId();

      int iet = (itt-1)/4 + 1;
      int ipt = (itt-1)%4 + 1;

      int ismt = Numbers::iSM( id );

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      if ( meIntegrityTTId[ismt-1] ) meIntegrityTTId[ismt-1]->Fill(xiet, xipt);

    }

  } else {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int itt = id.towerId();

      int iet = (itt-1)/4 + 1;
      int ipt = (itt-1)%4 + 1;

      int ismt = Numbers::iSM( id );

      float xiet = iet - 0.5;
      float xipt = ipt - 0.5;

      if ( meIntegrityTTBlockSize[ismt-1] ) meIntegrityTTBlockSize[ismt-1]->Fill(xiet, xipt);

    }

  } else {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( id );

      int itt   = id.towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTId[ism-1] ) meIntegrityMemTTId[ism-1]->Fill(iTt,0);

    }

  } else {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( id );

      int itt   = id.towerId();
      float iTt = itt + 0.5 - 69;

      if ( meIntegrityMemTTBlockSize[ism-1] ) meIntegrityMemTTBlockSize[ism-1]->Fill(iTt,0);

    }

  } else {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids8;

  if ( e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( id );

      int chid = id.channelId();
      int ie = EBIntegrityTask::chMemAbscissa[chid-1];
      int ip = EBIntegrityTask::chMemOrdinate[chid-1];

      int itt = id.towerId();
      ie += (itt-69)*5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityMemChId[ism-1] ) meIntegrityMemChId[ism-1]->Fill(xie,xip);

    }

  } else {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++ idItr ) {

      EcalElectronicsId id = (*idItr);

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( id );

      int chid = id.channelId();
      int ie = EBIntegrityTask::chMemAbscissa[chid-1];
      int ip = EBIntegrityTask::chMemOrdinate[chid-1];

      int itt = id.towerId();
      ie += (itt-69)*5;

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( meIntegrityMemGain[ism-1] ) meIntegrityMemGain[ism-1]->Fill(xie,xip);

    }

  } else {

    LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection6_ << " not available";

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

