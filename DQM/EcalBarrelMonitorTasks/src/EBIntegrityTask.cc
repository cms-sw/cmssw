/*
 * \file EBIntegrityTask.cc
 *
 * $Date: 2011/08/30 09:30:32 $
 * $Revision: 1.86 $
 * \author G. Della Ricca
 *
 */

#include <iostream>
#include <sstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "FWCore/Utilities/interface/Exception.h"

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
    meIntegrityTowerId[i] = 0;
    meIntegrityBlockSize[i] = 0;
    meIntegrityMemChId[i] = 0;
    meIntegrityMemGain[i] = 0;
    meIntegrityMemTowerId[i] = 0;
    meIntegrityMemBlockSize[i] = 0;
  }
  meIntegrityErrorsByLumi = 0;

  ievt_ = 0;

}


EBIntegrityTask::~EBIntegrityTask(){

}

void EBIntegrityTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/IntegrityErrors");
    dqmStore_->rmdir(prefixME_ + "/IntegrityErrors");
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
    if ( meIntegrityTowerId[i] ) meIntegrityTowerId[i]->Reset();
    if ( meIntegrityBlockSize[i] ) meIntegrityBlockSize[i]->Reset();
    if ( meIntegrityMemChId[i] ) meIntegrityMemChId[i]->Reset();
    if ( meIntegrityMemGain[i] ) meIntegrityMemGain[i]->Reset();
    if ( meIntegrityMemTowerId[i] ) meIntegrityMemTowerId[i]->Reset();
    if ( meIntegrityMemBlockSize[i] ) meIntegrityMemBlockSize[i]->Reset();
  }
  if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Reset();

}

void EBIntegrityTask::setup(void){

  init_ = true;

  std::string name;
  std::string dir;

  if ( dqmStore_ ) {
    dir = prefixME_ + "/IntegrityErrors";
    dqmStore_->setCurrentFolder(dir);

    // checking when number of towers in data different than expected from header
    name = "IntegrityTask DCC size error EB";
    meIntegrityDCCSize = dqmStore_->book1D(name, name, 36, 1., 37.);
    for (int i = 0; i < 36; i++) {
      meIntegrityDCCSize->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    // checking the number of integrity errors in each DCC for each lumi
    // crystal integrity error is weighted by 1/1700
    // tower integrity error is weighted by 1/68
    // bin 0 contains the number of processed events in the lumi (for normalization)
    name = "IntegrityTask errors by lumi EB";
    meIntegrityErrorsByLumi = dqmStore_->book1D(name, name, 36, 1., 37.);
    meIntegrityErrorsByLumi->setLumiFlag();
    for (int i = 0; i < 36; i++) {
      meIntegrityErrorsByLumi->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    dqmStore_->setCurrentFolder(dir + "/Gain");
    dqmStore_->setCurrentFolder(dir + "/ChId");
    dqmStore_->setCurrentFolder(dir + "/GainSwitch");
    dqmStore_->setCurrentFolder(dir + "/MEMChId");
    dqmStore_->setCurrentFolder(dir + "/MEMGain");
    dqmStore_->setCurrentFolder(dir + "/MEMBlockSize");
    dqmStore_->setCurrentFolder(dir + "/MEMTowerId");
    dqmStore_->setCurrentFolder(dir + "/BlockSize");
    dqmStore_->setCurrentFolder(dir + "/TowerId");

  }

}

void EBIntegrityTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if ( meIntegrityDCCSize ) dqmStore_->removeElement( meIntegrityDCCSize->getFullname() );
    meIntegrityDCCSize = 0;

    if ( meIntegrityErrorsByLumi ) dqmStore_->removeElement( meIntegrityErrorsByLumi->getFullname() );
    meIntegrityErrorsByLumi = 0;

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

  std::stringstream ss;
  std::string dir, name;
  MonitorElement *me(0);

  edm::Handle<EBDetIdCollection> ids1;

  if ( e.getByLabel(EBDetIdCollection1_, ids1) ) {

    dir = prefixME_ + "/IntegrityErrors/Gain/";

    for ( EBDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++idItr ) {

      EBDetId id = (*idItr);

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      ss.str("");
      ss << std::showpos << id.ieta() << std::noshowpos << " " << id.iphi();
      name = "IntegrityTask gain EB " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me) {
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection1_ << " not available";

  }

  edm::Handle<EBDetIdCollection> ids2;

  if ( e.getByLabel(EBDetIdCollection2_, ids2) ) {

    dir = prefixME_ + "/IntegrityErrors/ChId/";

    for ( EBDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++idItr ) {

      EBDetId id = (*idItr);

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      ss.str("");
      ss << std::showpos << id.ieta() << std::noshowpos << " " << id.iphi();
      name = "IntegrityTask ch id EB " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me) {
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection2_ << " not available";

  }

  edm::Handle<EBDetIdCollection> ids3;

  if ( e.getByLabel(EBDetIdCollection3_, ids3) ) {

    dir = prefixME_ + "/IntegrityErrors/GainSwitch/";

    for ( EBDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++idItr ) {

      EBDetId id = (*idItr);

      int ism = Numbers::iSM( id );
      float xism = ism + 0.5;

      ss.str("");
      ss << std::showpos << id.ieta() << std::noshowpos << " " << id.iphi();
      name = "IntegrityTask gain switch EB " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me) {
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xism);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EBDetIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    dir = prefixME_ + "/IntegrityErrors/TowerId/";

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int ismt = Numbers::iSM( *idItr );
      float xismt = ismt + 0.5;

      ss.str("");
      ss << idItr->dccId() << " " << idItr->towerId();
      name = "IntegrityTask tower id FE " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me) {
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);
      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xismt, 25.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    dir = prefixME_ + "/IntegrityErrors/BlockSize/";

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      int ismt = Numbers::iSM( *idItr );
      float xismt = ismt + 0.5;

      ss.str("");
      ss << idItr->dccId() << " " << idItr->towerId();
      name = "IntegrityTask block size FE " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me){
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

      if ( meIntegrityErrorsByLumi ) meIntegrityErrorsByLumi->Fill(xismt, 25.);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    dir = prefixME_ + "/IntegrityErrors/MEMTowerId/";

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      ss.str("");
      ss << idItr->dccId() << " " << idItr->towerId();
      name = "IntegrityTask MEM tower id FE " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me){
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    dir = prefixME_ + "/IntegrityErrors/MEMBlockSize/";

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      ss.str("");
      ss << idItr->dccId() << " " << idItr->towerId();
      name = "IntegrityTask MEM block size FE " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me){
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids8;

  if ( e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    dir = prefixME_ + "/IntegrityErrors/MEMChId/";

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      ss.str("");
      ss << idItr->dccId() << " " << idItr->towerId() << " " << idItr->stripId() << " " << idItr->xtalId();
      name = "IntegrityTask MEM ch id MEM " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me){
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

    }

  } else {

    edm::LogWarning("EBIntegrityTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    dir = prefixME_ + "/IntegrityErrors/MEMGain/";

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalBarrel ) continue;

      ss.str("");
      ss << idItr->dccId() << " " << idItr->towerId() << " " << idItr->stripId() << " " << idItr->xtalId();
      name = "IntegrityTask MEM gain MEM " + ss.str();
      me = dqmStore_->get(dir + name);
      if(!me){
	dqmStore_->setCurrentFolder(dir);
	me = dqmStore_->book1D(name, name, 1, 0., 1.);
      }
      if(me) me->Fill(0.5);

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

