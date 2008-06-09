/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2008/04/24 18:38:05 $
 * $Revision: 1.36 $
 * \author C. Bernet
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EETriggerTowerTask.h"

using namespace cms;
using namespace edm;
using namespace std;

const int EETriggerTowerTask::nTTEta = 20;
const int EETriggerTowerTask::nTTPhi = 20;
const int EETriggerTowerTask::nSM = 18;

EETriggerTowerTask::EETriggerTowerTask(const ParameterSet& ps) {

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meFlagsReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meFlagsEmul_);
  reserveArray(meEmulError_);
  reserveArray(meVetoEmulError_);
  reserveArray(meFlagEmulError_);

  realCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionReal");
  emulCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionEmul");

//   realModuleLabel_
//     = ps.getUntrackedParameter<string>("real_digis_moduleLabel", "ecalEBunpacker");
//   emulModuleLabel_
//     = ps.getUntrackedParameter<string>("emulated_digis_moduleLabel", "ecalTriggerPrimitiveDigis");
  outputFile_ = ps.getUntrackedParameter<string>("OutputRootFile", "");

  ostringstream  str;
  str<<"Module label for producer of REAL     digis: "<<realCollection_<<endl;
  str<<"Module label for producer of EMULATED digis: "<<emulCollection_<<endl;

  LogDebug("EETriggerTowerTask")<<str.str()<<endl;

}

EETriggerTowerTask::~EETriggerTowerTask(){

}

void EETriggerTowerTask::reserveArray( array1& array ) {

  array.reserve( nSM );
  array.resize( nSM, static_cast<MonitorElement*>(0) );

}

void EETriggerTowerTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETriggerTowerTask");
    dqmStore_->rmdir(prefixME_ + "/EETriggerTowerTask");
  }

  Numbers::initGeometry(c, false);

}

void EETriggerTowerTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EETriggerTowerTask::endRun(const Run& r, const EventSetup& c) {

}

void EETriggerTowerTask::reset(void) {

  for (int i = 0; i < 18; i++) {

    if ( meEtMapReal_[i] ) meEtMapReal_[i]->Reset();
    if ( meVetoReal_[i] ) meVetoReal_[i]->Reset();
    if ( meFlagsReal_[i] ) meFlagsReal_[i]->Reset();
    if ( meEtMapEmul_[i] ) meEtMapEmul_[i]->Reset();
    if ( meVetoEmul_[i] ) meVetoEmul_[i]->Reset();
    if ( meFlagsEmul_[i] ) meFlagsEmul_[i]->Reset();
    if ( meEmulError_[i] ) meEmulError_[i]->Reset();
    if ( meVetoEmulError_[i] ) meVetoEmulError_[i]->Reset();
    if ( meFlagEmulError_[i] ) meFlagEmulError_[i]->Reset();

  }

}

void EETriggerTowerTask::setup(void){

  init_ = true;

  if ( dqmStore_ ) {
    // dqmStore_->showDirStructure();

    setup( "Real Digis",
           (prefixME_ + "/EETriggerTowerTask").c_str(), false );

    setup( "Emulated Digis",
           (prefixME_ + "/EETriggerTowerTask/Emulated").c_str(), true);
  }
  else {
    LogError("EETriggerTowerTask")<<"Bad DQMStore, "
                                  <<"cannot book MonitorElements."<<endl;
  }
}

void EETriggerTowerTask::setup( const char* nameext,
                                const char* folder,
                                bool emulated ) {

  array1*  meEtMap = &meEtMapReal_;
  array1*  meVeto = &meVetoReal_;
  array1*  meFlags = &meFlagsReal_;

  if( emulated ) {
    meEtMap = &meEtMapEmul_;
    meVeto = &meVetoEmul_;
    meFlags= &meFlagsEmul_;
  }

  dqmStore_->setCurrentFolder(folder);

  static const unsigned namesize = 200;

  char histo[namesize];
  sprintf(histo, "EETTT Et map %s", nameext);
  string etMapName = histo;
  sprintf(histo, "EETTT FineGrainVeto %s", nameext);
  string fineGrainVetoName = histo;
  sprintf(histo, "EETTT Flags %s", nameext);
  string flagsName = histo;
  string emulErrorName = "EETTT EmulError";
  string emulFineGrainVetoErrorName = "EETTT EmulFineGrainVetoError";
  string emulFlagErrorName = "EETTT EmulFlagError";

  for (int i = 0; i < 18; i++) {

    string etMapNameSM = etMapName;
    etMapNameSM += " " + Numbers::sEE(i+1);

    (*meEtMap)[i] = dqmStore_->book3D(etMapNameSM.c_str(), etMapNameSM.c_str(),
                                50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                256, 0, 256.);
    (*meEtMap)[i]->setAxisTitle("jx", 1);
    (*meEtMap)[i]->setAxisTitle("jy", 2);
    dqmStore_->tag((*meEtMap)[i], i+1);

    string  fineGrainVetoNameSM = fineGrainVetoName;
    fineGrainVetoNameSM += " " + Numbers::sEE(i+1);

    (*meVeto)[i] = dqmStore_->book3D(fineGrainVetoNameSM.c_str(),
                               fineGrainVetoNameSM.c_str(),
                               50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                               50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                               2, 0., 2.);
    (*meVeto)[i]->setAxisTitle("jx", 1);
    (*meVeto)[i]->setAxisTitle("jy", 2);
    dqmStore_->tag((*meVeto)[i], i+1);

    string  flagsNameSM = flagsName;
    flagsNameSM += " " + Numbers::sEE(i+1);

    (*meFlags)[i] = dqmStore_->book3D(flagsNameSM.c_str(), flagsNameSM.c_str(),
                                50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                8, 0., 8.);
    (*meFlags)[i]->setAxisTitle("jx", 1);
    (*meFlags)[i]->setAxisTitle("jy", 2);
    dqmStore_->tag((*meFlags)[i], i+1);

    if(!emulated) {

      string  emulErrorNameSM = emulErrorName;
      emulErrorNameSM += " " + Numbers::sEE(i+1);

      meEmulError_[i] = dqmStore_->book2D(emulErrorNameSM.c_str(),
                                    emulErrorNameSM.c_str(),
                                    50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                    50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50. );
      meEmulError_[i]->setAxisTitle("jx", 1);
      meEmulError_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meEmulError_[i], i+1);

      string  emulFineGrainVetoErrorNameSM = emulFineGrainVetoErrorName;
      emulFineGrainVetoErrorNameSM += " " + Numbers::sEE(i+1);

      meVetoEmulError_[i] = dqmStore_->book3D(emulFineGrainVetoErrorNameSM.c_str(),
                                          emulFineGrainVetoErrorNameSM.c_str(),
                                          50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                          50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                          8, 0., 8.);
      meVetoEmulError_[i]->setAxisTitle("jx", 1);
      meVetoEmulError_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meVetoEmulError_[i], i+1);

      string  emulFlagErrorNameSM = emulFlagErrorName;
      emulFlagErrorNameSM += " " + Numbers::sEE(i+1);

      meFlagEmulError_[i] = dqmStore_->book3D(emulFlagErrorNameSM.c_str(),
                                          emulFlagErrorNameSM.c_str(),
                                          50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                          50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                          8, 0., 8.);
      meFlagEmulError_[i]->setAxisTitle("jx", 1);
      meFlagEmulError_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meFlagEmulError_[i], i+1);

    }
  }

}

void EETriggerTowerTask::cleanup(void) {

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if( !outputFile_.empty() ) dqmStore_->save( outputFile_.c_str() );

    dqmStore_->rmdir( prefixME_ + "/EETriggerTowerTask" );

  }

  init_ = false;

}

void EETriggerTowerTask::endJob(void){

  LogInfo("EETriggerTowerTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EETriggerTowerTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalTrigPrimDigiCollection> realDigis;

  if ( e.getByLabel(realCollection_, realDigis) ) {

    int neetpd = realDigis->size();
    LogDebug("EETriggerTowerTask")
      <<"event "
      <<ievt_
      <<" trigger primitive digi collection size: "
      <<neetpd;

    processDigis( realDigis,
                  meEtMapReal_,
                  meVetoReal_,
                  meFlagsReal_);

  } else {
    LogWarning("EETriggerTowerTask") << realCollection_ << " not available"; 
  }

  Handle<EcalTrigPrimDigiCollection> emulDigis;

  if ( e.getByLabel(emulCollection_, emulDigis) ) {

    processDigis( emulDigis,
                  meEtMapEmul_,
                  meVetoEmul_,
                  meFlagsEmul_,
                  realDigis);

  } else {
    LogWarning("EETriggerTowerTask") << emulCollection_ << " not available";
  }

}

void
EETriggerTowerTask::processDigis( const Handle<EcalTrigPrimDigiCollection>&
                                  digis,
                                  array1& meEtMap,
                                  array1& meVeto,
                                  array1& meFlags,
                                  const Handle<EcalTrigPrimDigiCollection>&
                                  compDigis ) {

  LogDebug("EETriggerTowerTask")<<"processing "<<meEtMap[0]->getName()<<endl;

  ostringstream  str;
  for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = digis->begin();
        tpdigiItr != digis->end(); ++tpdigiItr ) {

    EcalTriggerPrimitiveDigi data = (*tpdigiItr);
    EcalTrigTowerDetId idt = data.id();

    if ( Numbers::subDet( idt ) != EcalEndcap ) continue;

    int ismt = Numbers::iSM( idt );

    int itt = Numbers::iTT( idt );

    vector<DetId> crystals = Numbers::crystals( idt );

    for ( unsigned int i=0; i<crystals.size(); i++ ) {

    EEDetId id = crystals[i];

    int ix = id.ix();
    int iy = id.iy();

    if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

    float xix = ix-0.5;
    float xiy = iy-0.5;

    str<<"det id = "<<id.rawId()<<" "
       <<id<<" sm, tt, x, y "<<ismt<<" "<<itt<<" "<<ix<<" "<<iy<<endl;

    float xval;

    xval = data.compressedEt();
    if ( meEtMap[ismt-1] ) {
      meEtMap[ismt-1]->Fill(xix, xiy, xval);
    }
    else {
      LogError("EETriggerTowerTask")<<"histo does not exist "<<endl;
    }

    xval = 0.5 + data.fineGrain();
    if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xix, xiy, xval);

    xval = 0.5 + data.ttFlag();
    if ( meFlags[ismt-1] ) meFlags[ismt-1]->Fill(xix, xiy, xval);

    if( compDigis.isValid() ) {
      bool good = true;
      bool goodFlag = true;
      bool goodVeto = true;

      EcalTrigPrimDigiCollection::const_iterator compDigiItr = compDigis->find( idt.rawId() );
      if( compDigiItr != compDigis->end() ) {
        str<<"found corresponding digi! "<<*compDigiItr<<endl;
        if( data.compressedEt() != compDigiItr->compressedEt() ) {
          str<<"but it is different..."<<endl;
          good = false;
        }
        if( data.ttFlag() != compDigiItr->ttFlag() ) {
          str<<"but flag is different..."<<endl;
          goodFlag = false;
        }
        if( data.fineGrain() != compDigiItr->fineGrain() ) {
          str<<"but fine grain veto is different..."<<endl;
          goodVeto = false;
        }
      }
      else {
        good = false;
        goodFlag = false;
        goodVeto = false;
        str<<"could not find corresponding digi... "<<endl;
      }
      if(!good ) {
        if ( meEmulError_[ismt-1] ) meEmulError_[ismt-1]->Fill(xix, xiy);
      }
      if(!goodFlag) {
        float zval = data.ttFlag();
        if ( meFlagEmulError_[ismt-1] ) meFlagEmulError_[ismt-1]->Fill(xix, xiy, zval);
      }
      if(!goodVeto) {
        float zval = data.fineGrain();
        if ( meVetoEmulError_[ismt-1] ) meVetoEmulError_[ismt-1]->Fill(xix, xiy, zval);
      }
    }
    }
  }
  LogDebug("EETriggerTowerTask")<<str.str()<<endl;
}
