/*
 * \file EBTriggerTowerTask.cc
 *
 * $Date: 2009/08/23 20:59:52 $
 * $Revision: 1.86 $
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

  meEtSpectrumReal_ = 0;
  meEtSpectrumEmul_ = 0;
  meEtSpectrumEmulMax_ = 0;
  meEtBxReal_ = 0;
  meOccupancyBxReal_ = 0;
  
  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meEmulError_);
  reserveArray(meEmulMatch_);
  reserveArray(meVetoEmulError_);

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

  if ( meEtSpectrumReal_ ) meEtSpectrumReal_->Reset();
  if ( meEtSpectrumEmul_ ) meEtSpectrumEmul_->Reset();
  if ( meEtSpectrumEmulMax_ ) meEtSpectrumEmulMax_->Reset();
  if ( meEtBxReal_ ) meEtBxReal_->Reset();
  if ( meOccupancyBxReal_ ) meOccupancyBxReal_->Reset();
  
  for (int i = 0; i < 36; i++) {

    if ( meEtMapReal_[i] ) meEtMapReal_[i]->Reset();
    if ( meVetoReal_[i] ) meVetoReal_[i]->Reset();
    if ( meEtMapEmul_[i] ) meEtMapEmul_[i]->Reset();
    if ( meVetoEmul_[i] ) meVetoEmul_[i]->Reset();
    if ( meEmulError_[i] ) meEmulError_[i]->Reset();
    if ( meEmulMatch_[i] ) meEmulMatch_[i]->Reset();
    if ( meVetoEmulError_[i] ) meVetoEmulError_[i]->Reset();

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

  if( emulated ) {
    meEtMap = &meEtMapEmul_;
    meVeto = &meVetoEmul_;
  }

  dqmStore_->setCurrentFolder(folder);

  char histo[200];

  if(!emulated) {
    sprintf(histo, "EBTTT Et spectrum %s", nameext);
    meEtSpectrumReal_ = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumReal_->setAxisTitle("energy (ADC)", 1);

    double xbins[51];
    for ( int i=0; i<=11; i++ ) xbins[i] = i-1;  // begin of orbit
    // abort gap in presence of calibration: [3381-3500]
    // abort gap in absence of calibration: [3444-3500]
    // uing the wider abort gap always, start finer binning at bx=3371
    for ( int i=12; i<=22; i++) xbins[i] = 3371+i-12;
    // use 29 bins for the abort gap
    for ( int i=23; i<=51; i++) xbins[i] = 3382+(i-23)*6;

    sprintf(histo, "EBTTT Et vs bx %s", nameext);
    meEtBxReal_ = dqmStore_->bookProfile(histo, histo, 50, xbins, 256, 0, 256);
    meEtBxReal_->setAxisTitle("bunch crossing", 1);
    meEtBxReal_->setAxisTitle("energy (ADC)", 2);

    sprintf(histo, "EBTTT TP occupancy vs bx %s", nameext);
    meEtBxReal_ = dqmStore_->bookProfile(histo, histo, 50, xbins, 2448, 0, 2448);
    meEtBxReal_->setAxisTitle("bunch crossing", 1);
    meEtBxReal_->setAxisTitle("TP number", 2);

  } else {
    sprintf(histo, "EBTTT Et spectrum %s", nameext);
    meEtSpectrumEmul_ = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumEmul_->setAxisTitle("energy (ADC)", 1);

    sprintf(histo, "EBTTT Et spectrum %s max", nameext);
    meEtSpectrumEmulMax_ = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumEmulMax_->setAxisTitle("energy (ADC)", 1);
  }

  for (int i = 0; i < 36; i++) {

    sprintf(histo, "EBTTT Et map %s %s", nameext, Numbers::sEB(i+1).c_str());
    (*meEtMap)[i] = dqmStore_->bookProfile2D(histo, histo, nTTEta, 0, nTTEta, nTTPhi, 0, nTTPhi, 256, 0, 256.);
    (*meEtMap)[i]->setAxisTitle("ieta'", 1);
    (*meEtMap)[i]->setAxisTitle("iphi'", 2);
    dqmStore_->tag((*meEtMap)[i], i+1);

    if(!emulated) {

      sprintf(histo, "EBTTT EmulError %s", Numbers::sEB(i+1).c_str());
      meEmulError_[i] = dqmStore_->book2D(histo, histo, nTTEta, 0., nTTEta, nTTPhi, 0., nTTPhi );
      meEmulError_[i]->setAxisTitle("ieta'", 1);
      meEmulError_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meEmulError_[i], i+1);

      sprintf(histo, "EBTTT EmulMatch %s", Numbers::sEB(i+1).c_str());
      meEmulMatch_[i] = dqmStore_->book3D(histo, histo, nTTEta, 0., nTTEta, nTTPhi, 0., nTTPhi, 6, 0., 6.);
      meEmulMatch_[i]->setAxisTitle("ieta'", 1);
      meEmulMatch_[i]->setAxisTitle("iphi'", 2);
      meEmulMatch_[i]->setAxisTitle("TP timing", 3);
      dqmStore_->tag(meEmulMatch_[i], i+1);

      sprintf(histo, "EBTTT EmulFineGrainVetoError %s", Numbers::sEB(i+1).c_str());
      meVetoEmulError_[i] = dqmStore_->book2D(histo, histo, nTTEta, 0., nTTEta, nTTPhi, 0., nTTPhi);
      meVetoEmulError_[i]->setAxisTitle("ieta'", 1);
      meVetoEmulError_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meVetoEmulError_[i], i+1);

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
                  meVetoReal_);

  } else {
    LogWarning("EBTriggerTowerTask") << realCollection_ << " not available";
  }

  Handle<EcalTrigPrimDigiCollection> emulDigis;

  if ( e.getByLabel(emulCollection_, emulDigis) ) {

    processDigis( e,
                  emulDigis,
                  meEtMapEmul_,
                  meVetoEmul_,
                  realDigis);

  } else {
    LogWarning("EBTriggerTowerTask") << emulCollection_ << " not available";
  }

}

void
EBTriggerTowerTask::processDigis( const Event& e, const Handle<EcalTrigPrimDigiCollection>& digis,
                                  array1& meEtMap,
                                  array1& meVeto,
                                  const Handle<EcalTrigPrimDigiCollection>& compDigis ) {

  //  map<EcalTrigTowerDetId, int> crystalsInTower;
  int readoutCrystalsInTower[108][68];
    for (int itcc = 0; itcc < 108; itcc++) {
    for (int itt = 0; itt < 68; itt++) readoutCrystalsInTower[itcc][itt] = 0;
  }

  if( compDigis.isValid() ) {

    Handle<EBDigiCollection> crystalDigis;

    if ( e.getByLabel(EBDigiCollection_, crystalDigis) ) {

      for ( EBDigiCollection::const_iterator cDigiItr = crystalDigis->begin(); cDigiItr != crystalDigis->end(); ++cDigiItr ) {

        EBDetId id = cDigiItr->id();
        EcalTrigTowerDetId towid = id.tower();

        int itcc = Numbers::iTCC( towid );
        int itt = Numbers::iTT( towid );

        readoutCrystalsInTower[itcc-1][itt-1]++;

      }

    } else {
      LogWarning("EBTriggerTowerTask") << EBDigiCollection_ << " not available";
    }

  }

  int bx = e.bunchCrossing();
  int nTP = 0;

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

    int itt = Numbers::iTT( tpdigiItr->id() );
    int itcc = Numbers::iTCC( tpdigiItr->id() );

    float xvalEt = tpdigiItr->compressedEt();
    if ( meEtMap[ismt-1] ) meEtMap[ismt-1]->Fill(xiet, xipt, xvalEt);

    float xvalVeto = 0.5 + tpdigiItr->fineGrain();
    if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xiet, xipt, xvalVeto);

    if( compDigis.isValid() ) {

      if ( meEtSpectrumEmul_ ) meEtSpectrumEmul_->Fill( xvalEt );
      float maxEt = 0;
      for(int j=0; j<5; j++) {
        float EtTP = (*tpdigiItr)[j].compressedEt();
        if ( EtTP > maxEt ) maxEt = EtTP;
      }
      if ( meEtSpectrumEmulMax_ ) meEtSpectrumEmulMax_->Fill( maxEt );

      bool good = true;
      bool goodVeto = true;

      EcalTrigPrimDigiCollection::const_iterator compDigiItr = compDigis->find( tpdigiItr->id().rawId() );
      if( compDigiItr != compDigis->end() ) {

        if ( compDigiItr->compressedEt() > 0 ) nTP++;

        if ( meEtSpectrumReal_ ) meEtSpectrumReal_->Fill( compDigiItr->compressedEt() );

        if ( meEtBxReal_ && compDigiItr->compressedEt() > 0 ) meEtBxReal_->Fill( bx, compDigiItr->compressedEt() );

        if( tpdigiItr->compressedEt() != compDigiItr->compressedEt() ) {
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

        if(readoutCrystalsInTower[itcc-1][itt-1]==25 && compDigiItr->compressedEt()>0) {
          for(int j=0; j<6; j++) {
            if(matchSample[j]) meEmulMatch_[ismt-1]->Fill(xiet, xipt, j+0.5);
          }
        }

        if( tpdigiItr->fineGrain() != compDigiItr->fineGrain() ) {
          goodVeto = false;
        }
      }
      else {
        good = false;
        goodVeto = false;
      }
      if(!good ) {
        if ( meEmulError_[ismt-1] ) meEmulError_[ismt-1]->Fill(xiet, xipt);
      }
      if(!goodVeto) {
        if ( meVetoEmulError_[ismt-1] ) meVetoEmulError_[ismt-1]->Fill(xiet, xipt);
      }
    }

  }

  if ( meOccupancyBxReal_ ) meOccupancyBxReal_->Fill( bx, nTP );

}

