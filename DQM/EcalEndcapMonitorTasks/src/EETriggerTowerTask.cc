/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2012/04/27 13:46:16 $
 * $Revision: 1.82 $
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EETriggerTowerTask.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <iostream>

const int EETriggerTowerTask::nTTEta = 20;
const int EETriggerTowerTask::nTTPhi = 20;
const int EETriggerTowerTask::nSM = 18;

EETriggerTowerTask::EETriggerTowerTask(const edm::ParameterSet& ps) {

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  meEtSpectrumReal_[0] = 0;
  meEtSpectrumReal_[1] = 0;
  meEtSpectrumEmul_[0] = 0;
  meEtSpectrumEmul_[1] = 0;
  meEtSpectrumEmulMax_[0] = 0;
  meEtSpectrumEmulMax_[1] = 0;
  meEtBxReal_[0] = 0;
  meEtBxReal_[1] = 0;
  meOccupancyBxReal_[0] = 0;
  meOccupancyBxReal_[1] = 0;
  meTCCTimingCalo_[0] = 0;
  meTCCTimingCalo_[1] = 0;
  meTCCTimingMuon_[0] = 0;
  meTCCTimingMuon_[1] = 0;
  meEmulMatchIndex1D_[0] = 0;
  meEmulMatchIndex1D_[1] = 0;
  meEmulMatchMaxIndex1D_[0] = 0;
  meEmulMatchMaxIndex1D_[1] = 0;

  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meEmulError_);
  reserveArray(meEmulMatch_);
  reserveArray(meVetoEmulError_);

  realCollection_ =  ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollectionReal");
  emulCollection_ =  ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollectionEmul");
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  HLTResultsCollection_ = ps.getParameter<edm::InputTag>("HLTResultsCollection");

  HLTCaloHLTBit_ = ps.getUntrackedParameter<std::string>("HLTCaloHLTBit", "");
  HLTMuonHLTBit_ = ps.getUntrackedParameter<std::string>("HLTMuonHLTBit", "");

  outputFile_ = ps.getUntrackedParameter<std::string>("OutputRootFile", "");

  LogDebug("EETriggerTowerTask") << "REAL     digis: " << realCollection_;
  LogDebug("EETriggerTowerTask") << "EMULATED digis: " << emulCollection_;

}

EETriggerTowerTask::~EETriggerTowerTask(){

}

void EETriggerTowerTask::reserveArray( array1& array ) {

  array.reserve( nSM );
  array.resize( nSM, static_cast<MonitorElement*>(0) );

}

void EETriggerTowerTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETriggerTowerTask");
    dqmStore_->rmdir(prefixME_ + "/EETriggerTowerTask");
  }

}

void EETriggerTowerTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EETriggerTowerTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EETriggerTowerTask::reset(void) {

  for (int iside = 0; iside < 2; iside++) {
    if ( meEtSpectrumReal_[iside] ) meEtSpectrumReal_[iside]->Reset();
    if ( meEtSpectrumEmul_[iside] ) meEtSpectrumEmul_[iside]->Reset();
    if ( meEtSpectrumEmulMax_[iside] ) meEtSpectrumEmulMax_[iside]->Reset();
    if ( meEtBxReal_[iside] ) meEtBxReal_[iside]->Reset();
    if ( meOccupancyBxReal_[iside] ) meOccupancyBxReal_[iside]->Reset();
    if ( meTCCTimingCalo_[iside] ) meTCCTimingCalo_[iside]->Reset();
    if ( meTCCTimingMuon_[iside] ) meTCCTimingMuon_[iside]->Reset();
    if ( meEmulMatchIndex1D_[iside] ) meEmulMatchIndex1D_[iside]->Reset();
    if ( meEmulMatchMaxIndex1D_[iside] ) meEmulMatchMaxIndex1D_[iside]->Reset();
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
    edm::LogError("EETriggerTowerTask") << "Bad DQMStore, cannot book MonitorElements.";
  }
}

void EETriggerTowerTask::setup( std::string const &nameext,
                                std::string const &folder,
                                bool emulated ) {

  array1*  meEtMap = &meEtMapReal_;

  if ( emulated ) {
    meEtMap = &meEtMapEmul_;
  }

  dqmStore_->setCurrentFolder(folder);

  std::string name;

  if (!emulated) {
    name = "EETTT Et spectrum " + nameext + " EE -";
    meEtSpectrumReal_[0] = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumReal_[0]->setAxisTitle("energy (ADC)", 1);

    name = "EETTT Et spectrum " + nameext + " EE +";
    meEtSpectrumReal_[1] = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumReal_[1]->setAxisTitle("energy (ADC)", 1);

    name = "EETTT TP matching index EE -";
    meEmulMatchIndex1D_[0] = dqmStore_->book1D(name, name, 7, -1., 6.);
    meEmulMatchIndex1D_[0]->setAxisTitle("TP data matching emulator", 1);

    name = "EETTT TP matching index EE +";
    meEmulMatchIndex1D_[1] = dqmStore_->book1D(name, name, 7, -1., 6.);
    meEmulMatchIndex1D_[1]->setAxisTitle("TP data matching emulator", 1);

    name = "EETTT max TP matching index EE -";
    meEmulMatchMaxIndex1D_[0] = dqmStore_->book1D(name, name, 7, -1., 6.);
    meEmulMatchMaxIndex1D_[0]->setAxisTitle("Max TP data matching emulator", 1);

    name = "EETTT max TP matching index EE +";
    meEmulMatchMaxIndex1D_[1] = dqmStore_->book1D(name, name, 7, -1., 6.);
    meEmulMatchMaxIndex1D_[1]->setAxisTitle("Max TP data matching emulator", 1);

    double xbins[51];
    for ( int i=0; i<=11; i++ ) xbins[i] = i-1;  // begin of orbit
    // abort gap in presence of calibration: [3381-3500]
    // abort gap in absence of calibration: [3444-3500]
    // using the wider abort gap always, start finer binning at bx=3371
    for ( int i=12; i<=22; i++) xbins[i] = 3371+i-12;
    // use 29 bins for the abort gap
    for ( int i=23; i<=50; i++) xbins[i] = 3382+(i-23)*6;

    name = "EETTT Et vs bx " + nameext + " EE -";
    meEtBxReal_[0] = dqmStore_->bookProfile(name, name, 50, xbins, 256, 0, 256);
    meEtBxReal_[0]->setAxisTitle("bunch crossing", 1);
    meEtBxReal_[0]->setAxisTitle("energy (ADC)", 2);

    name = "EETTT Et vs bx " + nameext + " EE +";
    meEtBxReal_[1] = dqmStore_->bookProfile(name, name, 50, xbins, 256, 0, 256);
    meEtBxReal_[1]->setAxisTitle("bunch crossing", 1);
    meEtBxReal_[1]->setAxisTitle("energy (ADC)", 2);

    name = "EETTT TP occupancy vs bx " + nameext + " EE -";
    meOccupancyBxReal_[0] = dqmStore_->bookProfile(name, name, 50, xbins, 2448, 0, 2448);
    meOccupancyBxReal_[0]->setAxisTitle("bunch crossing", 1);
    meOccupancyBxReal_[0]->setAxisTitle("TP number", 2);

    name = "EETTT TP occupancy vs bx " + nameext + " EE +";
    meOccupancyBxReal_[1] = dqmStore_->bookProfile(name, name, 50, xbins, 2448, 0, 2448);
    meOccupancyBxReal_[1]->setAxisTitle("bunch crossing", 1);
    meOccupancyBxReal_[1]->setAxisTitle("TP number", 2);

    if ( HLTCaloHLTBit_ != "" ) {
      name = "EETTT TCC timing calo triggers " + nameext + " EE -";
      meTCCTimingCalo_[0] = dqmStore_->book2D(name, name, 36, 1, 37, 7, -1., 6.);
      meTCCTimingCalo_[0]->setAxisTitle("nTCC", 1);
      meTCCTimingCalo_[0]->setAxisTitle("TP data matching emulator", 2);

      name = "EETTT TCC timing calo triggers " + nameext + " EE +";
      meTCCTimingCalo_[1] = dqmStore_->book2D(name, name, 36, 73, 109, 7, -1., 6.);
      meTCCTimingCalo_[1]->setAxisTitle("nTCC", 1);
      meTCCTimingCalo_[1]->setAxisTitle("TP data matching emulator", 2);
    }

    if ( HLTMuonHLTBit_ != "" ) {
      name = "EETTT TCC timing muon triggers " + nameext + " EE -";
      meTCCTimingMuon_[0] = dqmStore_->book2D(name, name, 36, 1, 37, 7, -1., 6.);
      meTCCTimingMuon_[0]->setAxisTitle("nTCC", 1);
      meTCCTimingMuon_[0]->setAxisTitle("TP data matching emulator", 2);

      name = "EETTT TCC timing muon triggers " + nameext + " EE +";
      meTCCTimingMuon_[1] = dqmStore_->book2D(name, name, 36, 73, 109, 7, -1., 6.);
      meTCCTimingMuon_[1]->setAxisTitle("nTCC", 1);
      meTCCTimingMuon_[1]->setAxisTitle("TP data matching emulator", 2);
    }

  } else {
    name = "EETTT Et spectrum " + nameext + " EE -";
    meEtSpectrumEmul_[0] = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumEmul_[0]->setAxisTitle("energy (ADC)", 1);

    name = "EETTT Et spectrum " + nameext + " EE +";
    meEtSpectrumEmul_[1] = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumEmul_[1]->setAxisTitle("energy (ADC)", 1);

    name = "EETTT Et spectrum " + nameext + " max EE -";
    meEtSpectrumEmulMax_[0] = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumEmulMax_[0]->setAxisTitle("energy (ADC)", 1);

    name = "EETTT Et spectrum " + nameext + " max EE +";
    meEtSpectrumEmulMax_[1] = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumEmulMax_[1]->setAxisTitle("energy (ADC)", 1);
  }

  for (int i = 0; i < 18; i++) {

    name = "EETTT Et map " + nameext + " " + Numbers::sEE(i+1);
    (*meEtMap)[i] = dqmStore_->bookProfile2D(name, name,
                                             50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                             50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                             256, 0, 256.);
    (*meEtMap)[i]->setAxisTitle("ix", 1);
    if ( i+1 >= 1 && i+1 <= 9 ) (*meEtMap)[i]->setAxisTitle("101-ix", 1);
    (*meEtMap)[i]->setAxisTitle("iy", 2);
    dqmStore_->tag((*meEtMap)[i], i+1);

    if (!emulated) {

      name = "EETTT EmulError " + Numbers::sEE(i+1);
      meEmulError_[i] = dqmStore_->book2D(name, name,
                                          50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                          50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50. );
      meEmulError_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meEmulError_[i]->setAxisTitle("101-ix", 1);
      meEmulError_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meEmulError_[i], i+1);

      name = "EETTT EmulMatch " + Numbers::sEE(i+1);
      meEmulMatch_[i] = dqmStore_->book3D(name, name,
                                          50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                          50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
                                          6, 0., 6.);
      meEmulMatch_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meEmulMatch_[i]->setAxisTitle("101-ix", 1);
      meEmulMatch_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meEmulMatch_[i], i+1);

      name = "EETTT EmulFineGrainVetoError " + Numbers::sEE(i+1);
      meVetoEmulError_[i] = dqmStore_->book2D(name, name,
                                              50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
                                              50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meVetoEmulError_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meVetoEmulError_[i]->setAxisTitle("101-ix", 1);
      meVetoEmulError_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meVetoEmulError_[i], i+1);

    }
  }

}

void EETriggerTowerTask::cleanup(void) {

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if ( !outputFile_.empty() ) dqmStore_->save( outputFile_.c_str() );

    dqmStore_->rmdir( prefixME_ + "/EETriggerTowerTask" );

  }

  init_ = false;

}

void EETriggerTowerTask::endJob(void){

  edm::LogInfo("EETriggerTowerTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EETriggerTowerTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EcalTrigPrimDigiCollection> realDigis;

  if ( e.getByLabel(realCollection_, realDigis) ) {

    int neetpd = realDigis->size();
    LogDebug("EETriggerTowerTask") << "event " << ievt_ << " trigger primitive digi collection size: " << neetpd;

    processDigis( e,
                  realDigis,
                  meEtMapReal_,
                  meVetoReal_);

  } else {
    edm::LogWarning("EETriggerTowerTask") << realCollection_ << " not available";
  }

  edm::Handle<EcalTrigPrimDigiCollection> emulDigis;

  if ( e.getByLabel(emulCollection_, emulDigis) ) {

    edm::Handle<edm::TriggerResults> hltResults;

    if ( !e.getByLabel(HLTResultsCollection_, hltResults) ) {
      HLTResultsCollection_ = edm::InputTag(HLTResultsCollection_.label(), HLTResultsCollection_.instance(), "HLT");
    }

    if ( !e.getByLabel(HLTResultsCollection_, hltResults) ) {
      HLTResultsCollection_ = edm::InputTag(HLTResultsCollection_.label(), HLTResultsCollection_.instance(), "FU");
    }

    if ( e.getByLabel(HLTResultsCollection_, hltResults) ) {

      processDigis( e,
                    emulDigis,
                    meEtMapEmul_,
                    meVetoEmul_,
                    realDigis,
                    hltResults);

    } else {
      edm::LogWarning("EETriggerTowerTask") << HLTResultsCollection_ << " not available";
    }

  } else {
    edm::LogInfo("EETriggerTowerTask") << emulCollection_ << " not available";
  }

}

void
EETriggerTowerTask::processDigis( const edm::Event& e, const edm::Handle<EcalTrigPrimDigiCollection>& digis,
                                  array1& meEtMap,
                                  array1& meVeto,
                                  const edm::Handle<EcalTrigPrimDigiCollection>& compDigis,
                                  const edm::Handle<edm::TriggerResults> & hltResults ) {

  int bx = e.bunchCrossing();
  int nTP[2];
  nTP[0] = nTP[1] = 0;

  // indexes are: readoutCrystalsInTower[TCCId][iTT]
  int readoutCrystalsInTower[108][41];
  for (int itcc = 0; itcc < 108; itcc++) {
    for (int itt = 0; itt < 41; itt++) readoutCrystalsInTower[itcc][itt] = 0;
  }

  if ( compDigis.isValid() ) {

    edm::Handle<EEDigiCollection> crystalDigis;

    if ( e.getByLabel(EEDigiCollection_, crystalDigis) ) {

      for ( EEDigiCollection::const_iterator cDigiItr = crystalDigis->begin(); cDigiItr != crystalDigis->end(); ++cDigiItr ) {

        EEDetId id = cDigiItr->id();

        int ix = id.ix();
        int iy = id.iy();
        int ism = Numbers::iSM( id );
        int itcc = Numbers::iTCC( ism, EcalEndcap, ix, iy );
        int itt = Numbers::iTT( ism, EcalEndcap, ix, iy );

        readoutCrystalsInTower[itcc-1][itt-1]++;

      }

    } else {
      edm::LogWarning("EETriggerTowerTask") << EEDigiCollection_ << " not available";
    }

  }

  bool caloTrg = false;
  bool muonTrg = false;

  if ( hltResults.isValid() ) {

    int ntrigs = hltResults->size();
    if ( ntrigs!=0 ) {

      const edm::TriggerNames & triggerNames = e.triggerNames(*hltResults);

      for ( int itrig = 0; itrig != ntrigs; ++itrig ) {
        std::string trigName = triggerNames.triggerName(itrig);
        bool accept = hltResults->accept(itrig);

        if ( trigName == HLTCaloHLTBit_ ) caloTrg = accept;

        if ( trigName == HLTMuonHLTBit_ ) muonTrg = accept;

      }

    } else {
      edm::LogWarning("EBTriggerTowerTask") << " zero size trigger names in input TriggerResults";
    }

  }

  for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = digis->begin(); tpdigiItr != digis->end(); ++tpdigiItr ) {

    if ( Numbers::subDet( tpdigiItr->id() ) != EcalEndcap ) continue;

    int ismt = Numbers::iSM( tpdigiItr->id() );
    int itt = Numbers::iTT( tpdigiItr->id() );
    int itcc = Numbers::iTCC( tpdigiItr->id() );

    std::vector<DetId>* crystals = Numbers::crystals( tpdigiItr->id() );

    float xvalEt = tpdigiItr->compressedEt();
    float xvalVeto = 0.5 + tpdigiItr->fineGrain();

    bool good = true;
    bool goodVeto = true;

    int compDigiInterest = -1;

    bool matchSample[6];
    for (int j=0; j<6; j++) matchSample[j]=false;

    if ( compDigis.isValid() ) {

      if ( ismt >= 1 && ismt <= 9 ) {
        if ( meEtSpectrumEmul_[0] ) meEtSpectrumEmul_[0]->Fill( xvalEt );
      } else {
        if ( meEtSpectrumEmul_[1] ) meEtSpectrumEmul_[1]->Fill( xvalEt );
      }

      float maxEt = 0;
      int maxTPIndex = -1;
      for (int j=0; j<5; j++) {
        float EtTP = (*tpdigiItr)[j].compressedEt();
        if ( EtTP > maxEt ) {
          maxEt = EtTP;
          maxTPIndex = j+1;
        }
      }

      if ( ismt >= 1 && ismt <= 9 ) {
        if ( meEtSpectrumEmulMax_[0] ) meEtSpectrumEmulMax_[0]->Fill( maxEt );
        if ( meEmulMatchMaxIndex1D_[0] && maxEt > 0 ) meEmulMatchMaxIndex1D_[0]->Fill( maxTPIndex );
      } else {
        if ( meEtSpectrumEmulMax_[1] ) meEtSpectrumEmulMax_[1]->Fill( maxEt );
        if ( meEmulMatchMaxIndex1D_[1] && maxEt > 0 ) meEmulMatchMaxIndex1D_[1]->Fill( maxTPIndex );
      }

      EcalTrigPrimDigiCollection::const_iterator compDigiItr = compDigis->find( tpdigiItr->id().rawId() );
      if ( compDigiItr != compDigis->end() ) {
        int compDigiEt = compDigiItr->compressedEt();
        compDigiInterest = (compDigiItr->ttFlag() & 0x3);

        if ( ismt >= 1 && ismt <= 9 ) {
          if ( compDigiEt > 0 ) nTP[0]++;
          if ( meEtSpectrumReal_[0] ) meEtSpectrumReal_[0]->Fill( compDigiEt );
          if ( meEtBxReal_[0] && compDigiEt > 0 ) meEtBxReal_[0]->Fill( bx, compDigiEt );
        } else {
          if ( compDigiEt > 0 ) nTP[1]++;
          if ( meEtSpectrumReal_[1] ) meEtSpectrumReal_[1]->Fill( compDigiEt );
          if ( meEtBxReal_[1] && compDigiEt > 0 ) meEtBxReal_[1]->Fill( bx, compDigiEt );
        }

        // compare the 5 TPs with different time-windows
        // sample 0 means no match, 1-5: sample of the TP that matches
        matchSample[0]=false;
        bool matchedAny=false;

        for (int j=0; j<5; j++) {
          if ((*tpdigiItr)[j].compressedEt() == compDigiEt ) {
            matchSample[j+1]=true;
            matchedAny=true;
          } else {
            matchSample[j+1]=false;
          }
        }

        if (!matchedAny) matchSample[0]=true;

        // check if the tower has been readout completely and if it is medium or high interest
        if (readoutCrystalsInTower[itcc-1][itt-1] == int(crystals->size()) &&
            (compDigiInterest == 1 || compDigiInterest == 3) && compDigiEt > 0) {

          if ( tpdigiItr->compressedEt() != compDigiEt ) {
            good = false;
          }
          if ( tpdigiItr->fineGrain() != compDigiItr->fineGrain() ) {
            goodVeto = false;
          }

          for (int j=0; j<6; j++) {
            if (matchSample[j]) {

              int index = ( j==0 ) ? -1 : j;

              if ( ismt >= 1 && ismt <= 9 ) {
                meEmulMatchIndex1D_[0]->Fill(index+0.5);
              } else {
                meEmulMatchIndex1D_[1]->Fill(index+0.5);
              }

              for ( unsigned int i=0; i<crystals->size(); i++ ) {

                EEDetId id = (*crystals)[i];

                int ix = id.ix();
                int iy = id.iy();

                if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

                float xix = ix-0.5;
                float xiy = iy-0.5;

                meEmulMatch_[ismt-1]->Fill(xix, xiy, j+0.5);
                if ( ismt >= 1 && ismt <= 9 ) {
                  if ( meTCCTimingCalo_[0] && caloTrg ) meTCCTimingCalo_[0]->Fill( itcc, index+0.5 );
                  if ( meTCCTimingMuon_[0] && muonTrg ) meTCCTimingMuon_[0]->Fill( itcc, index+0.5 );
                } else {
                  if ( meTCCTimingCalo_[1] && caloTrg ) meTCCTimingCalo_[1]->Fill( itcc, index+0.5 );
                  if ( meTCCTimingMuon_[1] && muonTrg ) meTCCTimingMuon_[1]->Fill( itcc, index+0.5 );
                }

              } // loop on crystals

            }
          }

        } // check readout

      } else {
        good = false;
        goodVeto = false;
      }

      for ( unsigned int i=0; i<crystals->size(); i++ ) {

        EEDetId id = (*crystals)[i];

        int ix = id.ix();
        int iy = id.iy();

        if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

        float xix = ix-0.5;
        float xiy = iy-0.5;

        if (!good ) {
          if ( meEmulError_[ismt-1] ) meEmulError_[ismt-1]->Fill(xix, xiy);
        }
        if (!goodVeto) {
          if ( meVetoEmulError_[ismt-1] ) meVetoEmulError_[ismt-1]->Fill(xix, xiy);
        }

      } // loop on crystals

    } // compDigis.isValid

    for ( unsigned int i=0; i<crystals->size(); i++ ) {

      EEDetId id = (*crystals)[i];

      int ix = id.ix();
      int iy = id.iy();

      if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

      float xix = ix-0.5;
      float xiy = iy-0.5;

      if ( meEtMap[ismt-1] ) meEtMap[ismt-1]->Fill(xix, xiy, xvalEt);
      if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xix, xiy, xvalVeto);

    } // loop on crystals

  } // loop on TP

  if ( meOccupancyBxReal_[0] ) meOccupancyBxReal_[0]->Fill( bx, nTP[0] );
  if ( meOccupancyBxReal_[1] ) meOccupancyBxReal_[1]->Fill( bx, nTP[1] );

}

