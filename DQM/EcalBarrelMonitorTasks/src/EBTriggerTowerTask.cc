/*
 * \file EBTriggerTowerTask.cc
 *
 * $Date: 2011/11/01 20:44:55 $
 * $Revision: 1.108 $
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBTriggerTowerTask.h"
#include "FWCore/Common/interface/TriggerNames.h"

EBTriggerTowerTask::EBTriggerTowerTask(const edm::ParameterSet& ps) {

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEtSpectrumReal_ = 0;
  meEtSpectrumEmul_ = 0;
  meEtSpectrumEmulMax_ = 0;
  meEtBxReal_ = 0;
  meOccupancyBxReal_ = 0;
  meTCCTimingCalo_ = 0;
  meTCCTimingMuon_ = 0;
  meEmulMatchMaxIndex1D_ = 0;

  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meEmulError_);
  reserveArray(meEmulMatch_);
  reserveArray(meVetoEmulError_);

  realCollection_ =  ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollectionReal");
  emulCollection_ =  ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollectionEmul");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  HLTResultsCollection_ = ps.getParameter<edm::InputTag>("HLTResultsCollection");

  HLTCaloHLTBit_ = ps.getUntrackedParameter<std::string>("HLTCaloHLTBit", "");
  HLTMuonHLTBit_ = ps.getUntrackedParameter<std::string>("HLTMuonHLTBit", "");

  outputFile_ = ps.getUntrackedParameter<std::string>("OutputRootFile", "");

  LogDebug("EBTriggerTowerTask") << "REAL     digis: " << realCollection_;
  LogDebug("EBTriggerTowerTask") << "EMULATED digis: " << emulCollection_;

  ievt_ = 0;
}

EBTriggerTowerTask::~EBTriggerTowerTask(){

}

void EBTriggerTowerTask::reserveArray( array1& array ) {

  array.resize( 36, static_cast<MonitorElement*>(0) );

}

void EBTriggerTowerTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives");
    dqmStore_->rmdir(prefixME_ + "/TriggerPrimitives");
  }

}

void EBTriggerTowerTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBTriggerTowerTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBTriggerTowerTask::reset(void) {

  if ( meEtSpectrumReal_ ) meEtSpectrumReal_->Reset();
  if ( meEtSpectrumEmul_ ) meEtSpectrumEmul_->Reset();
  if ( meEtSpectrumEmulMax_ ) meEtSpectrumEmulMax_->Reset();
  if ( meEtBxReal_ ) meEtBxReal_->Reset();
  if ( meOccupancyBxReal_ ) meOccupancyBxReal_->Reset();
  if ( meTCCTimingCalo_ ) meTCCTimingCalo_->Reset();
  if ( meTCCTimingMuon_ ) meTCCTimingMuon_->Reset();
  if ( meEmulMatchMaxIndex1D_ ) meEmulMatchMaxIndex1D_->Reset();

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

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives");

    std::string name;

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/Et");

    name = "TrigPrimTask Et 1D EB";
    meEtSpectrumReal_ = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumReal_->setAxisTitle("energy (ADC)", 1);

    for (int i = 0; i < 36; i++) {
      name = "TrigPrimTask Et " + Numbers::sEB(i+1);
      meEtMapReal_[i] = dqmStore_->bookProfile2D(name, name, 17, 0, 85, 4, 0, 20, 256, 0, 256.);
      meEtMapReal_[i]->setAxisTitle("ieta'", 1);
      meEtMapReal_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meEtMapReal_[i], i+1);
    }

    double binEdges[] = {1., 271., 541., 892., 1162., 1432., 1783., 2053., 2323., 2674., 2944., 3214., 3446., 3490., 3491., 3565.};
    int nBXbins(sizeof(binEdges)/sizeof(double) - 1);
    name = "TrigPrimTask Et vs BX EB";
    meEtBxReal_ = dqmStore_->bookProfile(name, name, nBXbins, binEdges, 256, 0, 256);
    meEtBxReal_->setAxisTitle("bunch crossing", 1);
    meEtBxReal_->setAxisTitle("energy (ADC)", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/Et/Emulation");

    name = "TrigPrimTask emul Et 1D EB";
    meEtSpectrumEmul_ = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumEmul_->setAxisTitle("energy (ADC)", 1);

    name = "TrigPrimTask emul max Et 1D EB";
    meEtSpectrumEmulMax_ = dqmStore_->book1D(name, name, 256, 0., 256.);
    meEtSpectrumEmulMax_->setAxisTitle("energy (ADC)", 1);

    for (int i = 0; i < 36; i++) {
      name = "TrigPrimTask emul Et " +  Numbers::sEB(i+1);
      meEtMapEmul_[i] = dqmStore_->bookProfile2D(name, name, 17, 0, 85, 4, 0, 20, 256, 0, 256.);
      meEtMapEmul_[i]->setAxisTitle("ieta'", 1);
      meEtMapEmul_[i]->setAxisTitle("iphi'", 2);
      dqmStore_->tag(meEtMapEmul_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/EmulMatching");

    name = "TrigPrimTask emul max Et index EB";
    meEmulMatchMaxIndex1D_ = dqmStore_->book1D(name, name, 7, -1., 6.);
    meEmulMatchMaxIndex1D_->setAxisTitle("Max data matching emulator", 1);

    for (int i = 0; i < 36; i++) {
      name = "TrigPrimTask matching index " + Numbers::sEB(i+1);
      meEmulMatch_[i] = dqmStore_->book2D(name, name, 68, 0., 68., 7, -1., 6.);
      meEmulMatch_[i]->setAxisTitle("itt", 1);
      meEmulMatch_[i]->setAxisTitle("TP index matching emulator", 2);
      dqmStore_->tag(meEmulMatch_[i], i+1);
    }

    if ( HLTCaloHLTBit_ != "" ) {
      name = "TrigPrimTask matching index calo triggers EB";
      meTCCTimingCalo_ = dqmStore_->book2D(name, name, 36, 37, 73, 7, -1., 6.);
      meTCCTimingCalo_->setAxisTitle("itcc", 1);
      meTCCTimingCalo_->setAxisTitle("TP index matching emulator", 2);
    }

    if ( HLTMuonHLTBit_ != "" ) {
      name = "TrigPrimTask matching index muon triggers EB";
      meTCCTimingMuon_ = dqmStore_->book2D(name, name, 36, 37, 73, 7, -1., 6.);
      meTCCTimingMuon_->setAxisTitle("itcc", 1);
      meTCCTimingMuon_->setAxisTitle("TP index matching emulator", 2);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives");

    name = "TrigPrimTask TP number vs BX EB";
    meOccupancyBxReal_ = dqmStore_->bookProfile(name, name, nBXbins, binEdges, 2448, 0, 2448);
    meOccupancyBxReal_->setAxisTitle("bunch crossing", 1);
    meOccupancyBxReal_->setAxisTitle("TP number", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/TriggerPrimitives/EmulationErrors");

  } else {
    edm::LogError("EBTriggerTowerTask") << "Bad DQMStore, cannot book MonitorElements.";
  }
}

void EBTriggerTowerTask::cleanup(void) {

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    if ( !outputFile_.empty() ) dqmStore_->save( outputFile_ );

    dqmStore_->rmdir( prefixME_ + "/TriggerPrimitives" );

  }

  init_ = false;

}

void EBTriggerTowerTask::endJob(void){

  edm::LogInfo("EBTriggerTowerTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBTriggerTowerTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<EcalTrigPrimDigiCollection> realDigis;

  if ( e.getByLabel(realCollection_, realDigis) ) {

    int nebtpd = realDigis->size();
    LogDebug("EBTriggerTowerTask") << "event " << ievt_ <<" trigger primitive digi collection size: " << nebtpd;

    processDigis( e,
                  realDigis,
                  meEtMapReal_,
                  meVetoReal_);

  } else {
    edm::LogWarning("EBTriggerTowerTask") << realCollection_ << " not available";
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
      edm::LogWarning("EBTriggerTowerTask") << HLTResultsCollection_ << " not available";
    }

  } else {
    edm::LogInfo("EBTriggerTowerTask") << emulCollection_ << " not available";
  }

}

void
EBTriggerTowerTask::processDigis( const edm::Event& e, const edm::Handle<EcalTrigPrimDigiCollection>& digis,
                                  array1& meEtMap,
                                  array1& meVeto,
                                  const edm::Handle<EcalTrigPrimDigiCollection>& compDigis,
                                  const edm::Handle<edm::TriggerResults> & hltResults) {

  int bx = e.bunchCrossing();
  int nTP = 0;

  //  map<EcalTrigTowerDetId, int> crystalsInTower;
  int readoutCrystalsInTower[108][68];
    for (int itcc = 0; itcc < 108; itcc++) {
    for (int itt = 0; itt < 68; itt++) readoutCrystalsInTower[itcc][itt] = 0;
  }

  if ( compDigis.isValid() ) {

    edm::Handle<EBDigiCollection> crystalDigis;

    if ( e.getByLabel(EBDigiCollection_, crystalDigis) ) {

      for ( EBDigiCollection::const_iterator cDigiItr = crystalDigis->begin(); cDigiItr != crystalDigis->end(); ++cDigiItr ) {

        EBDetId id = cDigiItr->id();
        EcalTrigTowerDetId towid = id.tower();

        int itcc = Numbers::iTCC( towid );
        int itt = Numbers::iTT( towid );

        readoutCrystalsInTower[itcc-1][itt-1]++;

      }

    } else {
      edm::LogWarning("EBTriggerTowerTask") << EBDigiCollection_ << " not available";
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

    if ( Numbers::subDet( tpdigiItr->id() ) != EcalBarrel ) continue;

    int ismt = Numbers::iSM( tpdigiItr->id() );

    int iet = std::abs(tpdigiItr->id().ieta());
    int ipt = tpdigiItr->id().iphi();

    // phi_tower: change the range from global to SM-local
    // phi==0 is in the middle of a SM
    ipt = ipt + 2;
    if ( ipt > 72 ) ipt = ipt - 72;
    ipt = (ipt-1)%4 + 1;

    // phi_tower: SM-local phi runs opposite to global in EB+
    if ( tpdigiItr->id().zside() > 0 ) ipt = 5 - ipt;

    int itt = Numbers::iTT( tpdigiItr->id() );
    int itcc = Numbers::iTCC( tpdigiItr->id() );

    float xvalEt = tpdigiItr->compressedEt();
    float xvalVeto = 0.5 + tpdigiItr->fineGrain();

    bool good = true;
    bool goodVeto = true;

    int compDigiInterest = -1;

    bool matchSample[6];
    for (int j=0; j<6; j++) matchSample[j]=false;

    if ( compDigis.isValid() ) {

      if ( meEtSpectrumEmul_ ) meEtSpectrumEmul_->Fill( xvalEt );

      float maxEt = 0;
      int maxTPIndex = -1;
      for (int j=0; j<5; j++) {
        float EtTP = (*tpdigiItr)[j].compressedEt();
        if ( EtTP > maxEt ) {
          maxEt = EtTP;
          maxTPIndex = j+1;
        }
      }

      if ( meEtSpectrumEmulMax_ ) meEtSpectrumEmulMax_->Fill( maxEt );
      if ( meEmulMatchMaxIndex1D_ && maxEt > 0 ) meEmulMatchMaxIndex1D_->Fill( maxTPIndex );

      EcalTrigPrimDigiCollection::const_iterator compDigiItr = compDigis->find( tpdigiItr->id().rawId() );
      if ( compDigiItr != compDigis->end() ) {
        int compDigiEt = compDigiItr->compressedEt();
        compDigiInterest = (compDigiItr->ttFlag() & 0x3);

        if ( compDigiEt > 0 ) nTP++;
        if ( meEtSpectrumReal_ ) meEtSpectrumReal_->Fill( compDigiEt );
        if ( meEtBxReal_ && compDigiEt > 0 ) meEtBxReal_->Fill( bx, compDigiEt );

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
        if (readoutCrystalsInTower[itcc-1][itt-1] == 25 &&
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

              meEmulMatch_[ismt-1]->Fill(itt - 0.5, index+0.5);

              if ( meTCCTimingCalo_ && caloTrg ) meTCCTimingCalo_->Fill( itcc, index+0.5 );
              if ( meTCCTimingMuon_ && muonTrg ) meTCCTimingMuon_->Fill( itcc, index+0.5 );

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

    float xiet = iet * 5 - 2;
    float xipt = ipt * 5 - 2;

    if ( meEtMap[ismt-1] ) meEtMap[ismt-1]->Fill(xiet, xipt, xvalEt);
    if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xiet, xipt, xvalVeto);

  } // loop on TP

  if ( meOccupancyBxReal_ ) meOccupancyBxReal_->Fill( bx, nTP );

}

