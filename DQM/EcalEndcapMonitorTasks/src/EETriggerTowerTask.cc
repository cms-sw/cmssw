/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2009/08/05 10:54:50 $
 * $Revision: 1.49 $
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

  meEtSpectrumReal_[0] = 0;
  meEtSpectrumReal_[1] = 0;
  meEtSpectrumEmul_[0] = 0;
  meEtSpectrumEmul_[1] = 0;
  meEtSpectrumEmulMax_[0] = 0;
  meEtSpectrumEmulMax_[1] = 0;

  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meEmulError_);
  reserveArray(meEmulMatch_);
  reserveArray(meVetoEmulError_);

  realCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionReal");
  emulCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionEmul");
  EEDigiCollection_ = ps.getParameter<InputTag>("EEDigiCollection");

  outputFile_ = ps.getUntrackedParameter<string>("OutputRootFile", "");

  LogDebug("EETriggerTowerTask") << "REAL     digis: " << realCollection_;
  LogDebug("EETriggerTowerTask") << "EMULATED digis: " << emulCollection_;

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

  for (int iside = 0; iside < 2; iside++) {
    if ( meEtSpectrumReal_[iside] ) meEtSpectrumReal_[iside]->Reset();
    if ( meEtSpectrumEmul_[iside] ) meEtSpectrumEmul_[iside]->Reset();
    if ( meEtSpectrumEmulMax_[iside] ) meEtSpectrumEmulMax_[iside]->Reset();
  }

  for (int i = 0; i < 18; i++) {

    if ( meEtMapReal_[i] ) meEtMapReal_[i]->Reset();
    if ( meVetoReal_[i] ) meVetoReal_[i]->Reset();
    if ( meEtMapEmul_[i] ) meEtMapEmul_[i]->Reset();
    if ( meVetoEmul_[i] ) meVetoEmul_[i]->Reset();
    if ( meEmulError_[i] ) meEmulError_[i]->Reset();
    if ( meEmulMatch_[i] ) meEmulMatch_[i]->Reset();
    if ( meVetoEmulError_[i] ) meVetoEmulError_[i]->Reset();

  }

}

void EETriggerTowerTask::setup(void){

  init_ = true;

  if ( dqmStore_ ) {
    setup( "Real Digis",
           (prefixME_ + "/EETriggerTowerTask").c_str(), false );

    setup( "Emulated Digis",
           (prefixME_ + "/EETriggerTowerTask/Emulated").c_str(), true);
  }
  else {
    LogError("EETriggerTowerTask") << "Bad DQMStore, cannot book MonitorElements.";
  }
}

void EETriggerTowerTask::setup( const char* nameext,
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
    sprintf(histo, "EETTT Et spectrum %s EE -", nameext);
    meEtSpectrumReal_[0] = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumReal_[0]->setAxisTitle("energy (ADC)", 1);

    sprintf(histo, "EETTT Et spectrum %s EE +", nameext);
    meEtSpectrumReal_[1] = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumReal_[1]->setAxisTitle("energy (ADC)", 1);

  } else {
    sprintf(histo, "EETTT Et spectrum %s EE -", nameext);
    meEtSpectrumEmul_[0] = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumEmul_[0]->setAxisTitle("energy (ADC)", 1);

    sprintf(histo, "EETTT Et spectrum %s EE +", nameext);
    meEtSpectrumEmul_[1] = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumEmul_[1]->setAxisTitle("energy (ADC)", 1);

    sprintf(histo, "EETTT Et spectrum %s max EE -", nameext);
    meEtSpectrumEmulMax_[0] = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumEmulMax_[0]->setAxisTitle("energy (ADC)", 1);

    sprintf(histo, "EETTT Et spectrum %s max EE +", nameext);
    meEtSpectrumEmulMax_[1] = dqmStore_->book1D(histo, histo, 256, 0., 256.);
    meEtSpectrumEmulMax_[1]->setAxisTitle("energy (ADC)", 1);
  }

  for (int i = 0; i < 18; i++) {

    sprintf(histo, "EETTT Et map %s %s", nameext, Numbers::sEE(i+1).c_str());
    (*meEtMap)[i] = dqmStore_->bookProfile2D(histo, histo,
                                             50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                             50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                             256, 0, 256.);
    (*meEtMap)[i]->setAxisTitle("jx", 1);
    (*meEtMap)[i]->setAxisTitle("jy", 2);
    dqmStore_->tag((*meEtMap)[i], i+1);

    if(!emulated) {

      sprintf(histo, "EETTT EmulError %s", Numbers::sEE(i+1).c_str());
      meEmulError_[i] = dqmStore_->book2D(histo, histo,
                                          50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                          50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50. );
      meEmulError_[i]->setAxisTitle("jx", 1);
      meEmulError_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meEmulError_[i], i+1);

      sprintf(histo, "EETTT EmulMatch %s", Numbers::sEE(i+1).c_str());
      meEmulMatch_[i] = dqmStore_->book3D(histo, histo,
                                          50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                          50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                          6, 0., 6.);
      meEmulMatch_[i]->setAxisTitle("jx'", 1);
      meEmulMatch_[i]->setAxisTitle("jy'", 2);
      dqmStore_->tag(meEmulMatch_[i], i+1);

      sprintf(histo, "EETTT EmulFineGrainVetoError %s", Numbers::sEE(i+1).c_str());
      meVetoEmulError_[i] = dqmStore_->book2D(histo, histo,
                                              50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                              50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meVetoEmulError_[i]->setAxisTitle("jx", 1);
      meVetoEmulError_[i]->setAxisTitle("jy", 2);
      dqmStore_->tag(meVetoEmulError_[i], i+1);

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
    LogDebug("EETriggerTowerTask") << "event " << ievt_ << " trigger primitive digi collection size: " << neetpd;

    processDigis( e,
                  realDigis,
                  meEtMapReal_,
                  meVetoReal_);

  } else {
    LogWarning("EETriggerTowerTask") << realCollection_ << " not available";
  }

  Handle<EcalTrigPrimDigiCollection> emulDigis;

  if ( e.getByLabel(emulCollection_, emulDigis) ) {

    processDigis( e,
                  emulDigis,
                  meEtMapEmul_,
                  meVetoEmul_,
                  realDigis);

  } else {
    LogWarning("EETriggerTowerTask") << emulCollection_ << " not available";
  }

}

void
EETriggerTowerTask::processDigis( const Event& e, const Handle<EcalTrigPrimDigiCollection>& digis,
                                  array1& meEtMap,
                                  array1& meVeto,
                                  const Handle<EcalTrigPrimDigiCollection>& compDigis ) {

  int readoutCrystalsInTower[41];
  for(int i=0; i<41; i++) readoutCrystalsInTower[i] = 0;

  Handle<EEDigiCollection> crystalDigis;

  if ( e.getByLabel(EEDigiCollection_, crystalDigis) ) {

    for ( EEDigiCollection::const_iterator cDigiItr = crystalDigis->begin(); cDigiItr != crystalDigis->end(); ++cDigiItr ) {

      EEDetId id = cDigiItr->id();

      int ix = id.ix();
      int iy = id.iy();
      int ism = Numbers::iSM( id );
      int itt = Numbers::iTT( ism, EcalEndcap, ix, iy );

      readoutCrystalsInTower[itt-1]++;

    }

  } else {
    LogWarning("EETriggerTowerTask") << EEDigiCollection_ << " not available";
  }

  bool validCompDigis = false;
  if( compDigis.isValid() ) validCompDigis = true;

  for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = digis->begin(); tpdigiItr != digis->end(); ++tpdigiItr ) {

    if ( Numbers::subDet( tpdigiItr->id() ) != EcalEndcap ) continue;

    int ismt = Numbers::iSM( tpdigiItr->id() );
    int itt = Numbers::iTT( tpdigiItr->id() );

    float xvalEt = tpdigiItr->compressedEt();
    float xvalVeto = 0.5 + tpdigiItr->fineGrain();

    bool good = true;
    bool goodVeto = true;
    bool matchSample[6];
    for(int j=0; j<6; j++) matchSample[j] = false;
    bool matchedAny=false;

    int compDigiEt = -1;

    if( validCompDigis ) {

      if ( ismt >= 1 && ismt <= 9 ) {
        if ( meEtSpectrumEmul_[0] ) meEtSpectrumEmul_[0]->Fill( xvalEt );
      } else {
        if ( meEtSpectrumEmul_[1] ) meEtSpectrumEmul_[1]->Fill( xvalEt );
      }
      float maxEt = 0;
      for(int j=0; j<5; j++) {
        float EtTP = (*tpdigiItr)[j].compressedEt();
        if ( EtTP > maxEt ) maxEt = EtTP;
      }
      if ( ismt >= 1 && ismt <= 9 ) {
        if ( meEtSpectrumEmulMax_[0] ) meEtSpectrumEmulMax_[0]->Fill( maxEt );
      } else {
        if ( meEtSpectrumEmulMax_[1] ) meEtSpectrumEmulMax_[1]->Fill( maxEt );
      }

      EcalTrigPrimDigiCollection::const_iterator compDigiItr = compDigis->find( tpdigiItr->id().rawId() );
      if( compDigiItr != compDigis->end() ) {
        //        LogDebug("EETriggerTowerTask") << "found corresponding digi! "<< *compDigiItr;
        compDigiEt = compDigiItr->compressedEt();
        if ( ismt >= 1 && ismt <= 9 ) {
          if ( meEtSpectrumReal_[0] ) meEtSpectrumReal_[0]->Fill( compDigiEt );
        } else {
          if ( meEtSpectrumReal_[1] ) meEtSpectrumReal_[1]->Fill( compDigiEt );
        }
        if( tpdigiItr->compressedEt() != compDigiItr->compressedEt() ) {
          //          LogDebug("EETriggerTowerTask") << "but it is different...";
          good = false;
        }

        // compare the 5 TPs with different time-windows
        // sample 0 means no match, 1-5: sample of the TP that matches
        for(int j=0; j<5; j++) {
          if((*tpdigiItr)[j].compressedEt() == compDigiItr->compressedEt() ) {
            matchSample[j+1]=true;
            matchedAny=true;
          }
        }
        if(!matchedAny) matchSample[0]=true;
        if( tpdigiItr->fineGrain() != compDigiItr->fineGrain() ) {
          //          LogDebug("EETriggerTowerTask") << "but fine grain veto is different...";
          goodVeto = false;
        }
      } else {
        good = false;
        goodVeto = false;
      }
    }

    vector<DetId> crystals = Numbers::crystals( tpdigiItr->id() );
    
    int crystalsInTower = crystals.size();
    
    for ( unsigned int i=0; i<crystals.size(); i++ ) {
        
      EEDetId id = crystals[i];
        
      int ix = id.ix();
      int iy = id.iy();
      
      if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;
      
      float xix = ix-0.5;
      float xiy = iy-0.5;
      
      if ( meEtMap[ismt-1] ) meEtMap[ismt-1]->Fill(xix, xiy, xvalEt);

      if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xix, xiy, xvalVeto);

      if ( validCompDigis ) { 

        if(!good ) {
          if ( meEmulError_[ismt-1] ) meEmulError_[ismt-1]->Fill(xix, xiy);
        }
        if(!goodVeto) {
          if ( meVetoEmulError_[ismt-1] ) meVetoEmulError_[ismt-1]->Fill(xix, xiy);
        }

        if(readoutCrystalsInTower[itt-1]==crystalsInTower && compDigiEt > 0) {
          for(int j=0; j<6; j++) {
            if(matchSample[j]) meEmulMatch_[ismt-1]->Fill(xix, xiy, j+0.5);
          }
        }

      }

    } // loop over crustalsInTower
  } // loop over TPs
  
}

