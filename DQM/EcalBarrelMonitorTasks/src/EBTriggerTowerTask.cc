/*
 * \file EBTriggerTowerTask.cc
 *
 * $Date: 2008/12/08 08:01:50 $
 * $Revision: 1.82 $
 * \author C. Bernet
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBTriggerTowerTask.h"

using namespace cms;
using namespace edm;
using namespace std;

const int EBTriggerTowerTask::nTTEta = 17;
const int EBTriggerTowerTask::nTTPhi = 4;
const int EBTriggerTowerTask::nSM = 36;

EBTriggerTowerTask::EBTriggerTowerTask(const ParameterSet& ps) {

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meFlagsReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meFlagsEmul_);
  reserveArray(meEmulError_);
  reserveArray(meEmulMatch_);
  reserveArray(meVetoEmulError_);
  reserveArray(meFlagEmulError_);
  
  realCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionReal");
  emulCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionEmul");
  EBDigiCollection_ = ps.getParameter<InputTag>("EBDigiCollection");

  outputFile_ = ps.getUntrackedParameter<string>("OutputRootFile", "");

  LogDebug("EBTriggerTowerTask") << "REAL     digis: " << realCollection_;
  LogDebug("EBTriggerTowerTask") << "EMULATED digis: " << emulCollection_;

}

EBTriggerTowerTask::~EBTriggerTowerTask(){

}

void EBTriggerTowerTask::reserveArray( array1& array ) {

  array.reserve( nSM );
  array.resize( nSM, static_cast<MonitorElement*>(0) );

}

void EBTriggerTowerTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTriggerTowerTask");
    dqmStore_->rmdir(prefixME_ + "/EBTriggerTowerTask");
  }

  Numbers::initGeometry(c, false);

}

void EBTriggerTowerTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EBTriggerTowerTask::endRun(const Run& r, const EventSetup& c) {

}

void EBTriggerTowerTask::reset(void) {

  for (int i = 0; i < 36; i++) {

    if ( meEtMapReal_[i] ) meEtMapReal_[i]->Reset();
    if ( meVetoReal_[i] ) meVetoReal_[i]->Reset();
    if ( meFlagsReal_[i] ) meFlagsReal_[i]->Reset();
    if ( meEtMapEmul_[i] ) meEtMapEmul_[i]->Reset();
    if ( meVetoEmul_[i] ) meVetoEmul_[i]->Reset();
    if ( meFlagsEmul_[i] ) meFlagsEmul_[i]->Reset();
    if ( meEmulError_[i] ) meEmulError_[i]->Reset();
    if ( meEmulMatch_[i] ) meEmulMatch_[i]->Reset();
    if ( meVetoEmulError_[i] ) meVetoEmulError_[i]->Reset();
    if ( meFlagEmulError_[i] ) meFlagEmulError_[i]->Reset();

  }

}

void EBTriggerTowerTask::setup(void){

  init_ = true;

  if ( dqmStore_ ) {
    setup( "Real Digis",
           (prefixME_ + "/EBTriggerTowerTask").c_str(), false );

    setup( "Emulated Digis",
           (prefixME_ + "/EBTriggerTowerTask/Emulated").c_str(), true);
  }
  else {
    LogError("EBTriggerTowerTask") << "Bad DQMStore, cannot book MonitorElements.";
  }
}

void EBTriggerTowerTask::setup( const char* nameext,
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
  sprintf(histo, "EBTTT Et map %s", nameext);
  string etMapName = histo;
  sprintf(histo, "EBTTT FineGrainVeto %s", nameext);
  string fineGrainVetoName = histo;
  sprintf(histo, "EBTTT Flags %s", nameext);
  string flagsName = histo;
  string emulErrorName = "EBTTT EmulError";
  string emulMatchName = "EBTTT EmulMatch";
  string emulFineGrainVetoErrorName = "EBTTT EmulFineGrainVetoError";
  string emulFlagErrorName = "EBTTT EmulFlagError";


  for (int i = 0; i < 36; i++) {
    
    string etMapNameSM = etMapName;
    etMapNameSM += " " + Numbers::sEB(i+1);

    (*meEtMap)[i] = dqmStore_->bookProfile2D(etMapNameSM.c_str(), etMapNameSM.c_str(),
                                               nTTEta, 0, nTTEta,
                                               nTTPhi, 0, nTTPhi,
                                               256, 0, 256.);
    (*meEtMap)[i]->setAxisTitle("ieta'", 1);
    (*meEtMap)[i]->setAxisTitle("iphi'", 2);
    dqmStore_->tag((*meEtMap)[i], i+1);
    
    string fineGrainVetoNameSM = fineGrainVetoName;
    fineGrainVetoNameSM += " " + Numbers::sEB(i+1);

    (*meVeto)[i] = dqmStore_->book3D(fineGrainVetoNameSM.c_str(),
                               fineGrainVetoNameSM.c_str(),
                               nTTEta, 0, nTTEta,
                               nTTPhi, 0, nTTPhi,
                               2, 0., 2.);
    (*meVeto)[i]->setAxisTitle("ieta'", 1);
    (*meVeto)[i]->setAxisTitle("iphi'", 2);
    dqmStore_->tag((*meVeto)[i], i+1);

    string flagsNameSM = flagsName;
    flagsNameSM += " " + Numbers::sEB(i+1);

    (*meFlags)[i] = dqmStore_->book3D(flagsNameSM.c_str(), flagsNameSM.c_str(),
                                nTTEta, 0, nTTEta,
                                nTTPhi, 0, nTTPhi,
                                8, 0., 8.);
    (*meFlags)[i]->setAxisTitle("ieta'", 1);
    (*meFlags)[i]->setAxisTitle("iphi'", 2);
    dqmStore_->tag((*meFlags)[i], i+1);

    if(!emulated) {

      string emulErrorNameSM = emulErrorName;
      emulErrorNameSM += " " + Numbers::sEB(i+1);

      meEmulError_[i] = dqmStore_->book2D(emulErrorNameSM.c_str(),
                                    emulErrorNameSM.c_str(),
                                    nTTEta, 0., nTTEta,
                                    nTTPhi, 0., nTTPhi );
      meEmulError_[i]->setAxisTitle("ieta'", 1);
      meEmulError_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meEmulError_[i], i+1);

      string emulMatchNameSM = emulMatchName;
      emulMatchNameSM += " " + Numbers::sEB(i+1);

      meEmulMatch_[i] = dqmStore_->book3D(emulMatchNameSM.c_str(), emulMatchNameSM.c_str(),
                                          nTTEta, 0., nTTEta,
                                          nTTPhi, 0., nTTPhi,
                                          6, 0., 6.);
      meEmulMatch_[i]->setAxisTitle("ieta'", 1);
      meEmulMatch_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meEmulMatch_[i], i+1);

      string emulFineGrainVetoErrorNameSM = emulFineGrainVetoErrorName;
      emulFineGrainVetoErrorNameSM += " " + Numbers::sEB(i+1);

      meVetoEmulError_[i] = dqmStore_->book3D(emulFineGrainVetoErrorNameSM.c_str(),
                                          emulFineGrainVetoErrorNameSM.c_str(),
                                          nTTEta, 0., nTTEta,
                                          nTTPhi, 0., nTTPhi,
                                          8, 0., 8.);
      meVetoEmulError_[i]->setAxisTitle("ieta'", 1);
      meVetoEmulError_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meVetoEmulError_[i], i+1);

      string emulFlagErrorNameSM = emulFlagErrorName;
      emulFlagErrorNameSM += " " + Numbers::sEB(i+1);

      meFlagEmulError_[i] = dqmStore_->book3D(emulFlagErrorNameSM.c_str(),
                                          emulFlagErrorNameSM.c_str(),
                                          nTTEta, 0., nTTEta,
                                          nTTPhi, 0., nTTPhi,
                                          8, 0., 8.);
      meFlagEmulError_[i]->setAxisTitle("ieta'", 1);
      meFlagEmulError_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meFlagEmulError_[i], i+1);

    }
  }

}

void EBTriggerTowerTask::cleanup(void) {

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if( !outputFile_.empty() ) dqmStore_->save( outputFile_.c_str() );

    dqmStore_->rmdir( prefixME_ + "/EBTriggerTowerTask" );

  }

  init_ = false;

}

void EBTriggerTowerTask::endJob(void){

  LogInfo("EBTriggerTowerTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBTriggerTowerTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalTrigPrimDigiCollection> realDigis;

  if ( e.getByLabel(realCollection_, realDigis) ) {

    int nebtpd = realDigis->size();
    LogDebug("EBTriggerTowerTask") << "event " << ievt_ <<" trigger primitive digi collection size: " << nebtpd;

    processDigis( e,
                  realDigis,
                  meEtMapReal_,
                  meVetoReal_,
                  meFlagsReal_);

  } else {
    LogWarning("EBTriggerTowerTask") << realCollection_ << " not available";
  }

  Handle<EcalTrigPrimDigiCollection> emulDigis;

  if ( e.getByLabel(emulCollection_, emulDigis) ) {

    processDigis( e,
                  emulDigis,
                  meEtMapEmul_,
                  meVetoEmul_,
                  meFlagsEmul_,
                  realDigis);

  } else {
    LogWarning("EBTriggerTowerTask") << emulCollection_ << " not available";
  }

}

void
EBTriggerTowerTask::processDigis( const Event& e, const Handle<EcalTrigPrimDigiCollection>& digis,
                                  array1& meEtMap,
                                  array1& meVeto,
                                  array1& meFlags,
                                  const Handle<EcalTrigPrimDigiCollection>& compDigis ) {

  map<EcalTrigTowerDetId, int> crystalsInTower;

  if( compDigis.isValid() ) {

    Handle<EBDigiCollection> crystalDigis;

    if ( e.getByLabel(EBDigiCollection_, crystalDigis) ) {

      for ( EBDigiCollection::const_iterator cDigiItr = crystalDigis->begin(); cDigiItr != crystalDigis->end(); ++cDigiItr ) {

        EBDetId id = cDigiItr->id();
        EcalTrigTowerDetId towid = id.tower();

        map<EcalTrigTowerDetId, int>::const_iterator itrTower = crystalsInTower.find(towid);

        if( itrTower==crystalsInTower.end() ) crystalsInTower.insert(std::make_pair(towid,1));
        else crystalsInTower[towid]++;

      }

    } else {
      LogWarning("EBTriggerTowerTask") << EBDigiCollection_ << " not available";
    }

  }

  for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = digis->begin(); tpdigiItr != digis->end(); ++tpdigiItr ) {

    if ( Numbers::subDet( tpdigiItr->id() ) != EcalBarrel ) continue;

    int ismt = Numbers::iSM( tpdigiItr->id() );

    int iet = abs(tpdigiItr->id().ieta());
    int ipt = tpdigiItr->id().iphi();

    // phi_tower: change the range from global to SM-local
    // phi==0 is in the middle of a SM
    ipt = ipt + 2;
    if ( ipt > 72 ) ipt = ipt - 72;
    ipt = (ipt-1)%4 + 1;

    // phi_tower: SM-local phi runs opposite to global in EB+
    if ( tpdigiItr->id().zside() > 0 ) ipt = 5 - ipt;

    float xiet = iet-0.5;
    float xipt = ipt-0.5;

//    LogDebug("EBTriggerTowerTask") << "det id = "
//                                   << tpdigiItr->id().rawId() << " "
//                                   << "sm, tt, ieta, iphi "
//                                   << ismt << " " << itt << " "
//                                   << iet << " " << ipt;

    float xval;

    xval = tpdigiItr->compressedEt();
    if ( meEtMap[ismt-1] ) meEtMap[ismt-1]->Fill(xiet, xipt, xval);

    xval = 0.5 + tpdigiItr->fineGrain();
    if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xiet, xipt, xval);

    xval = 0.5 + tpdigiItr->ttFlag();
    if ( meFlags[ismt-1] ) meFlags[ismt-1]->Fill(xiet, xipt, xval);

    if( compDigis.isValid() ) {

      // count the number of readout crystals / TT
      // do do the match emul-real only if ncry/TT=25
      int nReadoutCrystals=crystalsInTower[tpdigiItr->id()];

      bool good = true;
      bool goodFlag = true;
      bool goodVeto = true;

      EcalTrigPrimDigiCollection::const_iterator compDigiItr = compDigis->find( tpdigiItr->id().rawId() );
      if( compDigiItr != compDigis->end() ) {
//        LogDebug("EBTriggerTowerTask") << "found corresponding digi! " << *compDigiItr;
        if( tpdigiItr->compressedEt() != compDigiItr->compressedEt() ) {
//          LogDebug("EBTriggerTowerTask") << "but it is different...";
          good = false;
        }

        // compare the 5 TPs with different time-windows
        // sample 0 means no match, 1-5: sample of the TP that matches
        bool matchSample[6];
        for(int j=0; j<6; j++) matchSample[j] = false;
        bool matchedAny=false;

        for(int j=0; j<5; j++) {
          if((*tpdigiItr)[j].compressedEt() == compDigiItr->compressedEt() ) {
            matchSample[j+1]=true;
            matchedAny=true;
          }
        }
        if(!matchedAny) matchSample[0]=true;

        if(nReadoutCrystals==25 && compDigiItr->compressedEt()>0) {
          for(int j=0; j<6; j++) {
            if(matchSample[j]) meEmulMatch_[ismt-1]->Fill(xiet, xipt, j+0.5);
          }
        }

        if( tpdigiItr->ttFlag() != compDigiItr->ttFlag() ) {
//          LogDebug("EBTriggerTowerTask") << "but flag is different...";
          goodFlag = false;
        }
        if( tpdigiItr->fineGrain() != compDigiItr->fineGrain() ) {
//          LogDebug("EBTriggerTowerTask") << "but fine grain veto is different...";
          goodVeto = false;
        }
      }
      else {
        good = false;
        goodFlag = false;
        goodVeto = false;
//        LogDebug("EBTriggerTowerTask") << "could not find corresponding digi...";
      }
      if(!good ) {
        if ( meEmulError_[ismt-1] ) meEmulError_[ismt-1]->Fill(xiet, xipt);
      }
      if(!goodFlag) {
        float zval = tpdigiItr->ttFlag();
        if ( meFlagEmulError_[ismt-1] ) meFlagEmulError_[ismt-1]->Fill(xiet, xipt, zval);
      }
      if(!goodVeto) {
        float zval = tpdigiItr->fineGrain();
        if ( meVetoEmulError_[ismt-1] ) meVetoEmulError_[ismt-1]->Fill(xiet, xipt, zval);
      }
    }
  }
}

