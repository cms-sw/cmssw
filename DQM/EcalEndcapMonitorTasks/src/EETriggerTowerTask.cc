/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2011/11/01 20:44:55 $
 * $Revision: 1.79 $
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

  ievt_ = 0;

  LogDebug("EETriggerTowerTask") << "REAL     digis: " << realCollection_;
  LogDebug("EETriggerTowerTask") << "EMULATED digis: " << emulCollection_;

}

EETriggerTowerTask::~EETriggerTowerTask(){

}

void EETriggerTowerTask::reserveArray( array1& array ) {

  array.resize( nSM, static_cast<MonitorElement*>(0) );

}

void EETriggerTowerTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives");
    dqmStore_->rmdir(prefixME_ + "/TriggerPrimitives");
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

  if(dqmStore_){

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives");

    std::string name;
    std::string subdet[2] = {"EE-", "EE+"};

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/Et");
    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      name = "TrigPrimTask Et 1D " + subdet[iSubdet];
      meEtSpectrumReal_[iSubdet] = dqmStore_->book1D(name, name, 256, 0., 256.);
      meEtSpectrumReal_[iSubdet]->setAxisTitle("energy (ADC)", 1);
    }

    for (int i = 0; i < 18; i++) {
      name = "TrigPrimTask Et " + Numbers::sEE(i+1);
      meEtMapReal_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 0, 256.);
      meEtMapReal_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meEtMapReal_[i]->setAxisTitle("101-ix", 1);
      meEtMapReal_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meEtMapReal_[i], i+1);
    }

    double binEdges[] = {1., 271., 541., 892., 1162., 1432., 1783., 2053., 2323., 2674., 2944., 3214., 3446., 3490., 3491., 3565.};
    int nBXbins(sizeof(binEdges)/sizeof(double) - 1);
    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      name = "TrigPrimTask Et vs BX " + subdet[iSubdet];
      meEtBxReal_[iSubdet] = dqmStore_->bookProfile(name, name, nBXbins, binEdges, 256, 0, 256);
      meEtBxReal_[iSubdet]->setAxisTitle("bunch crossing", 1);
      meEtBxReal_[iSubdet]->setAxisTitle("energy (ADC)", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/Et/Emulation");

    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      name = "TrigPrimTask emul Et 1D " + subdet[iSubdet];
      meEtSpectrumEmul_[iSubdet] = dqmStore_->book1D(name, name, 256, 0., 256.);
      meEtSpectrumEmul_[iSubdet]->setAxisTitle("energy (ADC)", 1);

      name = "TrigPrimTask emul max Et 1D " + subdet[iSubdet];
      meEtSpectrumEmulMax_[iSubdet] = dqmStore_->book1D(name, name, 256, 0., 256.);
      meEtSpectrumEmulMax_[iSubdet]->setAxisTitle("energy (ADC)", 1);
    }

    for (int i = 0; i < 18; i++) {
      name = "TrigPrimTask emul Et " + Numbers::sEE(i+1);
      meEtMapEmul_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 0, 256.);
      meEtMapEmul_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meEtMapEmul_[i]->setAxisTitle("101-ix", 1);
      meEtMapEmul_[i]->setAxisTitle("iy", 2);
      dqmStore_->tag(meEtMapEmul_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/EmulMatching");

    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      name = "TrigPrimTask emul max Et index " + subdet[iSubdet];
      meEmulMatchMaxIndex1D_[iSubdet] = dqmStore_->book1D(name, name, 7, -1., 6.);
      meEmulMatchMaxIndex1D_[iSubdet]->setAxisTitle("Max data matching emulator", 1);
    }

    for (int i = 0; i < 18; i++) {
      name = "TrigPrimTask matching index " + Numbers::sEE(i+1);
      meEmulMatch_[i] = dqmStore_->book2D(name, name, 80, 0., 80., 7, -1., 6.);
      meEmulMatch_[i]->setAxisTitle("itt", 1);
      meEmulMatch_[i]->setAxisTitle("TP index matching emulator", 2);
      dqmStore_->tag(meEmulMatch_[i], i+1);
    }

    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      if ( HLTCaloHLTBit_ != "" ) {
	name = "TrigPrimTask matching index calo triggers " + subdet[iSubdet];
	meTCCTimingCalo_[iSubdet] = dqmStore_->book2D(name, name, 36, 1, 37, 7, -1., 6.);
	meTCCTimingCalo_[iSubdet]->setAxisTitle("itcc", 1);
	meTCCTimingCalo_[iSubdet]->setAxisTitle("TP index matching emulator", 2);
      }

      if ( HLTMuonHLTBit_ != "" ) {
	name = "TrigPrimTask matching index muon triggers " + subdet[iSubdet];
	meTCCTimingMuon_[iSubdet] = dqmStore_->book2D(name, name, 36, 1, 37, 7, -1., 6.);
	meTCCTimingMuon_[iSubdet]->setAxisTitle("itcc", 1);
	meTCCTimingMuon_[iSubdet]->setAxisTitle("TP index matching emulator", 2);
      }
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives");

    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      name = "TrigPrimTask TP number vs BX " + subdet[iSubdet];
      meOccupancyBxReal_[iSubdet] = dqmStore_->bookProfile(name, name, nBXbins, binEdges, 2448, 0, 2448);
      meOccupancyBxReal_[iSubdet]->setAxisTitle("bunch crossing", 1);
      meOccupancyBxReal_[iSubdet]->setAxisTitle("TP number", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/EmulationErrors");

  }
  else {
    edm::LogError("EETriggerTowerTask") << "Bad DQMStore, cannot book MonitorElements.";
  }
}

void EETriggerTowerTask::cleanup(void) {

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if ( !outputFile_.empty() ) dqmStore_->save( outputFile_ );

    dqmStore_->rmdir( prefixME_ + "/TriggerPrimitives" );

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

  std::stringstream ss;
  std::string dir, name;
  MonitorElement *me(0);

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

	      // each DCC histogram bins are aligned by (inner low phi), (inner high phi), (outer low phi), (outer high phi)
	      int xitt;
	      if(itcc <= 18){ //inner EE-
		xitt = ((itcc - 1) % 2) * 24 + itt;
	      }else if(itcc <= 36){ //outer EE-
		xitt = 48 + ((itcc - 1) % 2) * 16 + itt;
	      }else if(itcc <= 90){ //outer EE+
		xitt = 48 + ((itcc - 1) % 2) * 16 + itt;
	      }else{
		xitt = ((itcc - 1) % 2) * 24 + itt;
	      }

	      meEmulMatch_[ismt-1]->Fill(xitt - 0.5, index);

	      if ( ismt >= 1 && ismt <= 9 ) {
		if ( meTCCTimingCalo_[0] && caloTrg ) meTCCTimingCalo_[0]->Fill( itcc, index+0.5 );
		if ( meTCCTimingMuon_[0] && muonTrg ) meTCCTimingMuon_[0]->Fill( itcc, index+0.5 );
	      } else {
		if ( meTCCTimingCalo_[1] && caloTrg ) meTCCTimingCalo_[1]->Fill( itcc, index+0.5 );
		if ( meTCCTimingMuon_[1] && muonTrg ) meTCCTimingMuon_[1]->Fill( itcc, index+0.5 );
	      }

            }
          }

        } // check readout

      } else {
        good = false;
        goodVeto = false;
      }

      if (!good ) {
	ss.str("");
	ss << "TT " << itcc << " " << itt;
	dir = prefixME_ + "/TriggerPrimitives/EmulationErrors/Et/";
	name = "TrigPrimTask emulation Et mismatch " + ss.str();
	me = dqmStore_->get(dir + name);
	if(!me) {
	  dqmStore_->setCurrentFolder(dir);
	  me = dqmStore_->book1D(name, name, 1, 0., 1.);
	}
	if(me) me->Fill(0.5);
      }
      if (!goodVeto) {
	ss.str("");
	ss << "TT " << itcc << " " << itt;
	dir = prefixME_ + "/TriggerPrimitives/EmulationErrors/FineGrainBit/";
	name = "TrigPrimTask emulation FGbit mismatch " + ss.str();
	me = dqmStore_->get(dir + name);
	if(!me) {
	  dqmStore_->setCurrentFolder(dir);
	  me = dqmStore_->book1D(name, name, 1, 0., 1.);
	}
	if(me) me->Fill(0.5);
      }

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

