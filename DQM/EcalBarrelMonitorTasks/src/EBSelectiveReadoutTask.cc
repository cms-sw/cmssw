/*
 * \file EBSelectiveReadoutTask.cc
 *
 * $Date: 2012/04/27 13:46:03 $
 * $Revision: 1.58 $
 * \author P. Gras
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cassert>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBSelectiveReadoutTask.h"


#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"


EBSelectiveReadoutTask::EBSelectiveReadoutTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  // parameters...
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EBUnsuppressedDigiCollection_ = ps.getParameter<edm::InputTag>("EBUsuppressedDigiCollection");
  EBSRFlagCollection_ = ps.getParameter<edm::InputTag>("EBSRFlagCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  firstFIRSample_ = ps.getParameter<int>("ecalDccZs1stSample");

  useCondDb_ = ps.getParameter<bool>("configFromCondDB");

  if(!useCondDb_) configFirWeights(ps.getParameter<std::vector<double> >("dccWeights"));


  // histograms...
  EBTowerSize_ = 0;
  EBTTFMismatch_ = 0;
  EBDccEventSize_ = 0;
  EBDccEventSizeMap_ = 0;
  EBReadoutUnitForcedBitMap_ = 0;
  EBFullReadoutSRFlagMap_ = 0;
  EBFullReadoutSRFlagCount_ = 0;
  EBZeroSuppression1SRFlagMap_ = 0;
  EBHighInterestTriggerTowerFlagMap_ = 0;
  EBMediumInterestTriggerTowerFlagMap_ = 0;
  EBLowInterestTriggerTowerFlagMap_ = 0;
  EBTTFlags_ = 0;
  EBCompleteZSMap_ = 0;
  EBCompleteZSCount_ = 0;
  EBDroppedFRMap_ = 0;
  EBDroppedFRCount_ = 0;
  EBEventSize_ = 0;
  EBHighInterestPayload_ = 0;
  EBLowInterestPayload_ = 0;
  EBHighInterestZsFIR_ = 0;
  EBLowInterestZsFIR_ = 0;

  // initialize variable binning for DCC size...
  float ZSthreshold = 0.608; // kBytes of 1 TT fully readout
  float zeroBinSize = ZSthreshold / 20.;
  for(int i=0; i<20; i++) ybins[i] = i*zeroBinSize;
  for(int i=20; i<89; i++) ybins[i] = ZSthreshold * (i-19);
  for(int i=0; i<=36; i++) xbins[i] = i+1;

}

EBSelectiveReadoutTask::~EBSelectiveReadoutTask() {

}

void EBSelectiveReadoutTask::beginJob(void) {

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");
    dqmStore_->rmdir(prefixME_ + "/EBSelectiveReadoutTask");
  }

}

void EBSelectiveReadoutTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  for(int ietindex = 0; ietindex < 34; ietindex++ ) {
    for(int iptindex = 0; iptindex < 72; iptindex++ ) {
      nEvtFullReadout[iptindex][ietindex] = 0;
      nEvtZS1Readout[iptindex][ietindex] = 0;
      nEvtZSReadout[iptindex][ietindex] = 0;
      nEvtCompleteReadoutIfZS[iptindex][ietindex] = 0;
      nEvtDroppedReadoutIfFR[iptindex][ietindex] = 0;
      nEvtRUForced[iptindex][ietindex] = 0;
      nEvtAnyReadout[iptindex][ietindex] = 0;
      nEvtHighInterest[iptindex][ietindex] = 0;
      nEvtMediumInterest[iptindex][ietindex] = 0;
      nEvtLowInterest[iptindex][ietindex] = 0;
      nEvtAnyInterest[iptindex][ietindex] = 0;
    }
  }

  //getting selective readout configuration
  if(useCondDb_) {
    edm::ESHandle<EcalSRSettings> hSr;
    c.get<EcalSRSettingsRcd>().get(hSr);
    settings_ = hSr.product();
    std::vector<double> wsFromDB;

    std::vector<std::vector<float> > dccs = settings_->dccNormalizedWeights_;
    int nws = dccs.size();
    if(nws == 1) {
      for(std::vector<float>::const_iterator it = dccs[0].begin(); it != dccs[0].end(); it++) {
	wsFromDB.push_back(*it);
      }
    }
    else edm::LogWarning("EBSelectiveReadoutTask") << "DCC weight set is not exactly 1.";

    configFirWeights(wsFromDB);
  }

}

void EBSelectiveReadoutTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBSelectiveReadoutTask::reset(void) {

  if ( EBTowerSize_ ) EBTowerSize_->Reset();

  if ( EBTTFMismatch_ ) EBTTFMismatch_->Reset();

  if ( EBDccEventSize_ ) EBDccEventSize_->Reset();

  if ( EBDccEventSizeMap_ ) EBDccEventSizeMap_->Reset();

  if ( EBReadoutUnitForcedBitMap_ ) EBReadoutUnitForcedBitMap_->Reset();

  if ( EBFullReadoutSRFlagMap_ ) EBFullReadoutSRFlagMap_->Reset();

  if ( EBFullReadoutSRFlagCount_ ) EBFullReadoutSRFlagCount_->Reset();

  if ( EBZeroSuppression1SRFlagMap_ ) EBZeroSuppression1SRFlagMap_->Reset(); 

  if ( EBHighInterestTriggerTowerFlagMap_ ) EBHighInterestTriggerTowerFlagMap_->Reset();

  if ( EBMediumInterestTriggerTowerFlagMap_ ) EBMediumInterestTriggerTowerFlagMap_->Reset();

  if ( EBLowInterestTriggerTowerFlagMap_ ) EBLowInterestTriggerTowerFlagMap_->Reset();
  
  if ( EBTTFlags_ ) EBTTFlags_->Reset();

  if ( EBCompleteZSMap_ ) EBCompleteZSMap_->Reset();
  
  if ( EBCompleteZSCount_ ) EBCompleteZSCount_->Reset();

  if ( EBDroppedFRMap_ ) EBDroppedFRMap_->Reset();

  if ( EBDroppedFRCount_ ) EBDroppedFRCount_->Reset();

  if ( EBEventSize_ ) EBEventSize_->Reset();

  if ( EBHighInterestPayload_ ) EBHighInterestPayload_->Reset();

  if ( EBLowInterestPayload_ ) EBLowInterestPayload_->Reset();

  if ( EBHighInterestZsFIR_ ) EBHighInterestZsFIR_->Reset();

  if ( EBLowInterestZsFIR_ ) EBLowInterestZsFIR_->Reset();

}

void EBSelectiveReadoutTask::setup(void) {

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");

    name = "EBSRT tower event size";
    EBTowerSize_ = dqmStore_->bookProfile2D(name, name, 72, 0, 72, 34, -17, 17, 100, 0, 200, "s");
    EBTowerSize_->setAxisTitle("jphi", 1);
    EBTowerSize_->setAxisTitle("jeta", 2);

    name = "EBSRT TT flag mismatch";
    EBTTFMismatch_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBTTFMismatch_->setAxisTitle("jphi", 1);
    EBTTFMismatch_->setAxisTitle("jeta", 2);

    name = "EBSRT DCC event size";
    EBDccEventSize_ = dqmStore_->bookProfile(name, name, 36, 1, 37, 100, 0., 200., "s");
    EBDccEventSize_->setAxisTitle("event size (kB)", 2);
    for (int i = 0; i < 36; i++) {
      EBDccEventSize_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBSRT event size vs DCC";
    EBDccEventSizeMap_ = dqmStore_->book2D(name, name, 36, xbins, 88, ybins);
    EBDccEventSizeMap_->setAxisTitle("event size (kB)", 2);
    for (int i = 0; i < 36; i++) {
      EBDccEventSizeMap_->setBinLabel(i+1, Numbers::sEB(i+1), 1);
    }

    name = "EBSRT readout unit with SR forced";
    EBReadoutUnitForcedBitMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBReadoutUnitForcedBitMap_->setAxisTitle("jphi", 1);
    EBReadoutUnitForcedBitMap_->setAxisTitle("jeta", 2);
    EBReadoutUnitForcedBitMap_->setAxisTitle("rate", 3);

    name = "EBSRT full readout SR Flags";
    EBFullReadoutSRFlagMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBFullReadoutSRFlagMap_->setAxisTitle("jphi", 1);
    EBFullReadoutSRFlagMap_->setAxisTitle("jeta", 2);
    EBFullReadoutSRFlagMap_->setAxisTitle("rate", 3);

    name = "EBSRT full readout SR Flags Number";
    EBFullReadoutSRFlagCount_ = dqmStore_->book1D(name, name, 200, 0., 200.);
    EBFullReadoutSRFlagCount_->setAxisTitle("Readout Units number", 1);

    name = "EBSRT zero suppression 1 SR Flags";
    EBZeroSuppression1SRFlagMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBZeroSuppression1SRFlagMap_->setAxisTitle("jphi", 1);
    EBZeroSuppression1SRFlagMap_->setAxisTitle("jeta", 2);
    EBZeroSuppression1SRFlagMap_->setAxisTitle("rate", 3);

    name = "EBSRT high interest TT Flags";
    EBHighInterestTriggerTowerFlagMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBHighInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EBHighInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);
    EBHighInterestTriggerTowerFlagMap_->setAxisTitle("rate", 3);

    name = "EBSRT medium interest TT Flags";
    EBMediumInterestTriggerTowerFlagMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBMediumInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EBMediumInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);
    EBMediumInterestTriggerTowerFlagMap_->setAxisTitle("rate", 3);

    name = "EBSRT low interest TT Flags";
    EBLowInterestTriggerTowerFlagMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBLowInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EBLowInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);
    EBLowInterestTriggerTowerFlagMap_->setAxisTitle("rate", 3);

    name = "EBSRT TT Flags";
    EBTTFlags_ = dqmStore_->book1D(name, name, 8, 0., 8.);
    EBTTFlags_->setAxisTitle("TT Flag value", 1);

    name = "EBSRT ZS Flagged Fully Readout";
    EBCompleteZSMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBCompleteZSMap_->setAxisTitle("jphi", 1);
    EBCompleteZSMap_->setAxisTitle("jeta", 2);
    EBCompleteZSMap_->setAxisTitle("rate", 3);

    name = "EBSRT ZS Flagged Fully Readout Number";
    EBCompleteZSCount_ = dqmStore_->book1D(name, name, 20, 0., 20.);
    EBCompleteZSCount_->setAxisTitle("Readout Units number", 1);

    name = "EBSRT FR Flagged Dropped Readout";
    EBDroppedFRMap_ = dqmStore_->book2D(name, name, 72, 0, 72, 34, -17, 17);
    EBDroppedFRMap_->setAxisTitle("jphi", 1);
    EBDroppedFRMap_->setAxisTitle("jeta", 2);
    EBDroppedFRMap_->setAxisTitle("rate", 3);

    name = "EBSRT FR Flagged Dropped Readout Number";
    EBDroppedFRCount_ = dqmStore_->book1D(name, name, 20, 0., 20.);
    EBDroppedFRCount_->setAxisTitle("Readout Units number", 1);

    name = "EBSRT event size";
    EBEventSize_ = dqmStore_->book1D(name, name, 100, 0, 200);
    EBEventSize_->setAxisTitle("event size (kB)",1);

    name = "EBSRT high interest payload";
    EBHighInterestPayload_ =  dqmStore_->book1D(name, name, 100, 0, 200);
    EBHighInterestPayload_->setAxisTitle("event size (kB)",1);

    name = "EBSRT low interest payload";
    EBLowInterestPayload_ =  dqmStore_->book1D(name, name, 100, 0, 200);
    EBLowInterestPayload_->setAxisTitle("event size (kB)",1);

    name = "EBSRT high interest ZS filter output";
    EBHighInterestZsFIR_ = dqmStore_->book1D(name, name, 60, -30, 30);
    EBHighInterestZsFIR_->setAxisTitle("ADC counts*4",1);

    name = "EBSRT low interest ZS filter output";
    EBLowInterestZsFIR_ = dqmStore_->book1D(name, name, 60, -30, 30);
    EBLowInterestZsFIR_->setAxisTitle("ADC counts*4",1);

  }

}

void EBSelectiveReadoutTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");

    if ( EBTowerSize_ ) dqmStore_->removeElement( EBTowerSize_->getName() );
    EBTowerSize_ = 0;

    if ( EBTTFMismatch_ ) dqmStore_->removeElement( EBTTFMismatch_->getName() );
    EBTTFMismatch_ = 0;

    if ( EBDccEventSize_ ) dqmStore_->removeElement( EBDccEventSize_->getName() );
    EBDccEventSize_ = 0;

    if ( EBDccEventSizeMap_ ) dqmStore_->removeElement( EBDccEventSizeMap_->getName() );
    EBDccEventSizeMap_ = 0;

    if ( EBReadoutUnitForcedBitMap_ ) dqmStore_->removeElement( EBReadoutUnitForcedBitMap_->getName() );
    EBReadoutUnitForcedBitMap_ = 0;

    if ( EBFullReadoutSRFlagMap_ ) dqmStore_->removeElement( EBFullReadoutSRFlagMap_->getName() );
    EBFullReadoutSRFlagMap_ = 0;

    if ( EBFullReadoutSRFlagCount_ ) dqmStore_->removeElement( EBFullReadoutSRFlagCount_->getName() );
    EBFullReadoutSRFlagCount_ = 0;

    if ( EBFullReadoutSRFlagCount_ ) dqmStore_->removeElement( EBFullReadoutSRFlagCount_->getName() );
    EBFullReadoutSRFlagCount_ = 0;

    if ( EBHighInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EBHighInterestTriggerTowerFlagMap_->getName() );
    EBHighInterestTriggerTowerFlagMap_ = 0;

    if ( EBMediumInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EBMediumInterestTriggerTowerFlagMap_->getName() );
    EBMediumInterestTriggerTowerFlagMap_ = 0;

    if ( EBLowInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EBLowInterestTriggerTowerFlagMap_->getName() );
    EBLowInterestTriggerTowerFlagMap_ = 0;

    if ( EBTTFlags_ ) dqmStore_->removeElement( EBTTFlags_->getName() );
    EBTTFlags_ = 0;

    if ( EBCompleteZSMap_ ) dqmStore_->removeElement( EBCompleteZSMap_->getName() );
    EBCompleteZSMap_ = 0;

    if ( EBCompleteZSCount_ ) dqmStore_->removeElement( EBCompleteZSCount_->getName() );
    EBCompleteZSCount_ = 0;

    if ( EBDroppedFRMap_ ) dqmStore_->removeElement( EBDroppedFRMap_->getName() );
    EBDroppedFRMap_ = 0;

    if ( EBDroppedFRCount_ ) dqmStore_->removeElement( EBDroppedFRCount_->getName() );
    EBDroppedFRCount_ = 0;

    if ( EBEventSize_ ) dqmStore_->removeElement( EBEventSize_->getName() );
    EBEventSize_ = 0;

    if ( EBHighInterestPayload_ ) dqmStore_->removeElement( EBHighInterestPayload_->getName() );
    EBHighInterestPayload_ = 0;

    if ( EBLowInterestPayload_ ) dqmStore_->removeElement( EBLowInterestPayload_->getName() );
    EBLowInterestPayload_ = 0;

    if ( EBHighInterestZsFIR_ ) dqmStore_->removeElement( EBHighInterestZsFIR_->getName() );
    EBHighInterestZsFIR_ = 0;

    if ( EBLowInterestZsFIR_ ) dqmStore_->removeElement( EBLowInterestZsFIR_->getName() );
    EBLowInterestZsFIR_ = 0;

  }

  init_ = false;

}

void EBSelectiveReadoutTask::endJob(void){

  edm::LogInfo("EBSelectiveReadoutTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBSelectiveReadoutTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<FEDRawDataCollection> raw;
  if ( e.getByLabel(FEDRawDataCollection_, raw) ) {

    for ( int iDcc = 0; iDcc < nEBDcc; ++iDcc ) {

      EBDccEventSize_->Fill(iDcc+1, ((double)raw->FEDData(610+iDcc).size())/kByte );
      EBDccEventSizeMap_->Fill(iDcc+1, ((double)raw->FEDData(610+iDcc).size())/kByte);

    }

  } else {
    edm::LogWarning("EBSelectiveReadoutTask") << FEDRawDataCollection_ << " not available";
  }

  // Selective Readout Flags
  int nFRO, nCompleteZS, nDroppedFRO;
  nFRO = 0;
  nCompleteZS = 0;
  nDroppedFRO = 0;
  edm::Handle<EBSrFlagCollection> ebSrFlags;
  if ( e.getByLabel(EBSRFlagCollection_,ebSrFlags) ) {

    // Data Volume
    double aLowInterest=0;
    double aHighInterest=0;
    double aAnyInterest=0;

    edm::Handle<EBDigiCollection> ebDigis;
    if ( e.getByLabel(EBDigiCollection_ , ebDigis) ) {

      anaDigiInit();

      // channel status
      edm::ESHandle<EcalChannelStatus> pChannelStatus;
      c.get<EcalChannelStatusRcd>().get(pChannelStatus);
      const EcalChannelStatus* chStatus = pChannelStatus.product();  

      for (unsigned int digis=0; digis<ebDigis->size(); ++digis){
        EBDataFrame ebdf = (*ebDigis)[digis];
        EBDetId id = ebdf.id();
        EcalChannelStatusMap::const_iterator chit;
        chit = chStatus->getMap().find(id.rawId());
        uint16_t statusCode = 0;
        if( chit != chStatus->getMap().end() ) {
          EcalChannelStatusCode ch_code = (*chit);
          statusCode = ch_code.getStatusCode();
        }
        anaDigi(ebdf, *ebSrFlags, statusCode);
      }

      //low interest channels:
      aLowInterest = nEbLI_*bytesPerCrystal/kByte;
      EBLowInterestPayload_->Fill(aLowInterest);

      //low interest channels:
      aHighInterest = nEbHI_*bytesPerCrystal/kByte;
      EBHighInterestPayload_->Fill(aHighInterest);

      //any-interest channels:
      aAnyInterest = getEbEventSize(nEb_)/kByte;
      EBEventSize_->Fill(aAnyInterest);

      //event size by tower:
      for(int ietindex = 0; ietindex < 34; ietindex++ ) {
        for(int iptindex = 0; iptindex < 72; iptindex++ ) {

          float xiet = (ietindex < 17) ? ietindex + 0.5 : (16-ietindex) + 0.5;
          float xipt = iptindex + 0.5;

          double towerSize =  nCryTower[iptindex][ietindex] * bytesPerCrystal;
          EBTowerSize_->Fill(xipt, xiet, towerSize);

        }
      }
    } else {
      edm::LogWarning("EBSelectiveReadoutTask") << EBDigiCollection_ << " not available";
    }

    // initialize dcchs_ to mask disabled towers
    std::map< int, std::vector<short> > towersStatus;
    edm::Handle<EcalRawDataCollection> dcchs;

    if( e.getByLabel(FEDRawDataCollection_, dcchs) ) {
      for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {
        if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;
        int ism = Numbers::iSM( *dcchItr, EcalBarrel );
        towersStatus.insert(std::make_pair(ism, dcchItr->getFEStatus()));
      }
    }

    for ( EBSrFlagCollection::const_iterator it = ebSrFlags->begin(); it != ebSrFlags->end(); ++it ) {

      EcalTrigTowerDetId id = it->id();

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

      int iet = id.ieta();
      int ietindex = (iet>0) ? iet - 1 : 16 + std::abs(iet);
      // phi_tower: change the range from global to SM-local
      // phi==0 is in the middle of a SM
      int ipt = id.iphi() + 2;
      if ( ipt > 72 ) ipt = ipt - 72;
      int iptindex = ipt - 1;
      int ism = Numbers::iSM( id );
      int itt = Numbers::iTT( id );

      nEvtAnyReadout[iptindex][ietindex]++;

      int flag = it->value() & ~EcalSrFlag::SRF_FORCED_MASK;

      int status=0;
      if( towersStatus[ism].size() > 0 ) status = (towersStatus[ism])[itt];

      if(flag == EcalSrFlag::SRF_FULL) {
        nEvtFullReadout[iptindex][ietindex]++;
        nFRO++;
        if(nPerRu_[ism-1][itt-1] == 0) {
          if(status != 1) nEvtDroppedReadoutIfFR[iptindex][ietindex]++;
          nDroppedFRO++;
        }
      }

      if(flag == EcalSrFlag::SRF_ZS1) nEvtZS1Readout[iptindex][ietindex]++;

      if(it->value() & EcalSrFlag::SRF_FORCED_MASK) nEvtRUForced[iptindex][ietindex]++;

      if(flag == EcalSrFlag::SRF_ZS1 || flag == EcalSrFlag::SRF_ZS2) {
        nEvtZSReadout[iptindex][ietindex]++;
        if(nPerRu_[ism-1][itt-1] == getCrystalCount()) {
          if(status != 1) nEvtCompleteReadoutIfZS[iptindex][ietindex]++;
          nCompleteZS++;
        }
      }

    }
  } else {
    edm::LogWarning("EBSelectiveReadoutTask") << EBSRFlagCollection_ << " not available";
  }

  for(int ietindex = 0; ietindex < 34; ietindex++ ) {
    for(int iptindex = 0; iptindex < 72; iptindex++ ) {

      if(nEvtAnyReadout[iptindex][ietindex]) {

        float xiet = (ietindex < 17) ? ietindex + 0.5 : (16-ietindex) + 0.5;
        float xipt = iptindex + 0.5;

        float fraction = float(nEvtFullReadout[iptindex][ietindex]) / float(nEvtAnyReadout[iptindex][ietindex]);
        float error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[iptindex][ietindex]));

        TH2F *h2d = EBFullReadoutSRFlagMap_->getTH2F();

        int binet=0, binpt=0;

        if ( h2d ) {
          binpt = h2d->GetXaxis()->FindBin(xipt);
          binet = h2d->GetYaxis()->FindBin(xiet);
        }

        EBFullReadoutSRFlagMap_->setBinContent(binpt, binet, fraction);
        EBFullReadoutSRFlagMap_->setBinError(binpt, binet, error);


        fraction = float(nEvtZS1Readout[iptindex][ietindex]) / float(nEvtAnyReadout[iptindex][ietindex]);
        error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[iptindex][ietindex]));

        h2d = EBZeroSuppression1SRFlagMap_->getTH2F();

        if ( h2d ) {
          binpt = h2d->GetXaxis()->FindBin(xipt);
          binet = h2d->GetYaxis()->FindBin(xiet);
        }

        EBZeroSuppression1SRFlagMap_->setBinContent(binpt, binet, fraction);
        EBZeroSuppression1SRFlagMap_->setBinError(binpt, binet, error);


        fraction = float(nEvtRUForced[iptindex][ietindex]) / float(nEvtAnyReadout[iptindex][ietindex]);
        error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[iptindex][ietindex]));

        h2d = EBReadoutUnitForcedBitMap_->getTH2F();

        if ( h2d ) {
          binpt = h2d->GetXaxis()->FindBin(xipt);
          binet = h2d->GetYaxis()->FindBin(xiet);
        }

        EBReadoutUnitForcedBitMap_->setBinContent(binpt, binet, fraction);
        EBReadoutUnitForcedBitMap_->setBinError(binpt, binet, error);

        if( nEvtZSReadout[iptindex][ietindex] ) {
          fraction = float(nEvtCompleteReadoutIfZS[iptindex][ietindex]) / float(nEvtZSReadout[iptindex][ietindex]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[iptindex][ietindex]));

          h2d = EBCompleteZSMap_->getTH2F();
          
          if ( h2d ) {
            binpt = h2d->GetXaxis()->FindBin(xipt);
            binet = h2d->GetYaxis()->FindBin(xiet);
          }
          
          EBCompleteZSMap_->setBinContent(binpt, binet, fraction);
          EBCompleteZSMap_->setBinError(binpt, binet, error);
        }

        if( nEvtFullReadout[iptindex][ietindex] ) {
          fraction = float(nEvtDroppedReadoutIfFR[iptindex][ietindex]) / float(nEvtFullReadout[iptindex][ietindex]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[iptindex][ietindex]));
          
          h2d = EBDroppedFRMap_->getTH2F();
          
          if ( h2d ) {
            binpt = h2d->GetXaxis()->FindBin(xipt);
            binet = h2d->GetYaxis()->FindBin(xiet);
          }
          
          EBDroppedFRMap_->setBinContent(binpt, binet, fraction);
          EBDroppedFRMap_->setBinError(binpt, binet, error);
        }
      }

    }
  }

  EBFullReadoutSRFlagCount_->Fill( nFRO );
  EBCompleteZSCount_->Fill( nCompleteZS );
  EBDroppedFRCount_->Fill( nDroppedFRO );

  edm::Handle<EcalTrigPrimDigiCollection> TPCollection;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, TPCollection) ) {

    // Trigger Primitives
    EcalTrigPrimDigiCollection::const_iterator TPdigi;
    for (TPdigi = TPCollection->begin(); TPdigi != TPCollection->end(); ++TPdigi ) {

      if ( Numbers::subDet( TPdigi->id() ) != EcalBarrel ) continue;

      int iet = TPdigi->id().ieta();
      int ietindex = (iet>0) ? iet - 1 : 16 + std::abs(iet);
      // phi_tower: change the range from global to SM-local
      // phi==0 is in the middle of a SM
      int ipt = TPdigi->id().iphi() + 2;
      if ( ipt > 72 ) ipt = ipt - 72;
      int iptindex = ipt - 1;

      nEvtAnyInterest[iptindex][ietindex]++;

      if ( (TPdigi->ttFlag() & 0x3) == 0 ) nEvtLowInterest[iptindex][ietindex]++;

      if ( (TPdigi->ttFlag() & 0x3) == 1 ) nEvtMediumInterest[iptindex][ietindex]++;

      if ( (TPdigi->ttFlag() & 0x3) == 3 ) nEvtHighInterest[iptindex][ietindex]++;

      EBTTFlags_->Fill( TPdigi->ttFlag() );

      float xiet = (ietindex < 17) ? ietindex + 0.5 : (16-ietindex) + 0.5;
      float xipt = iptindex + 0.5;

      if ( ((TPdigi->ttFlag() & 0x3) == 1 || (TPdigi->ttFlag() & 0x3) == 3)
           && nCryTower[iptindex][ietindex] != 25 ) EBTTFMismatch_->Fill(xipt, xiet);

    }
  } else {
    edm::LogWarning("EBSelectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }

  for(int ietindex = 0; ietindex < 34; ietindex++ ) {
    for(int iptindex = 0; iptindex < 72; iptindex++ ) {

      if(nEvtAnyInterest[iptindex][ietindex]) {

        float xiet = (ietindex < 17) ? ietindex + 0.5 : (16-ietindex) + 0.5;
        float xipt = iptindex + 0.5;

        float fraction = float(nEvtHighInterest[iptindex][ietindex]) / float(nEvtAnyInterest[iptindex][ietindex]);
        float error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[iptindex][ietindex]));

        TH2F *h2d = EBHighInterestTriggerTowerFlagMap_->getTH2F();

        int binet=0, binpt=0;

        if ( h2d ) {
          binpt = h2d->GetXaxis()->FindBin(xipt);
          binet = h2d->GetYaxis()->FindBin(xiet);
        }

        EBHighInterestTriggerTowerFlagMap_->setBinContent(binpt, binet, fraction);
        EBHighInterestTriggerTowerFlagMap_->setBinError(binpt, binet, error);


        fraction = float(nEvtMediumInterest[iptindex][ietindex]) / float(nEvtAnyInterest[iptindex][ietindex]);
        error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[iptindex][ietindex]));

        h2d = EBMediumInterestTriggerTowerFlagMap_->getTH2F();

        if ( h2d ) {
          binpt = h2d->GetXaxis()->FindBin(xipt);
          binet = h2d->GetYaxis()->FindBin(xiet);
        }

        EBMediumInterestTriggerTowerFlagMap_->setBinContent(binpt, binet, fraction);
        EBMediumInterestTriggerTowerFlagMap_->setBinError(binpt, binet, error);


        fraction = float(nEvtLowInterest[iptindex][ietindex]) / float(nEvtAnyInterest[iptindex][ietindex]);
        error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[iptindex][ietindex]));

        h2d = EBLowInterestTriggerTowerFlagMap_->getTH2F();

        if ( h2d ) {
          binpt = h2d->GetXaxis()->FindBin(xipt);
          binet = h2d->GetYaxis()->FindBin(xiet);
        }

        EBLowInterestTriggerTowerFlagMap_->setBinContent(binpt, binet, fraction);
        EBLowInterestTriggerTowerFlagMap_->setBinError(binpt, binet, error);

      }

    }
  }


}

void EBSelectiveReadoutTask::anaDigi(const EBDataFrame& frame, const EBSrFlagCollection& srFlagColl, uint16_t statusCode){

  EBDetId id = frame.id();

  bool barrel = (id.subdetId()==EcalBarrel);

  if(barrel){
    ++nEb_;

    int ieta = id.ieta();
    int iphi = id.iphi();

    int iEta0 = iEta2cIndex(ieta);
    int iPhi0 = iPhi2cIndex(iphi);
    if(!ebRuActive_[iEta0/ebTtEdge][iPhi0/ebTtEdge]){
      ++nRuPerDcc_[dccNum(id)-1];
      ebRuActive_[iEta0/ebTtEdge][iPhi0/ebTtEdge] = true;
    }

    EcalTrigTowerDetId towid = id.tower();
    int iet = towid.ieta();
    int ietindex = (iet>0) ? iet - 1 : 16 + std::abs(iet);
    // phi_tower: change the range from global to SM-local
    // phi==0 is in the middle of a SM
    int ipt = towid.iphi() + 2;
    if ( ipt > 72 ) ipt = ipt - 72;
    int iptindex = ipt - 1;
    
    int ism = Numbers::iSM( id );
    int itt = Numbers::iTT( towid );

    nCryTower[iptindex][ietindex]++;
    
    EBSrFlagCollection::const_iterator srf = srFlagColl.find(readOutUnitOf(id));
    
    if(srf == srFlagColl.end()){
      return;
    }
    
    bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
                         == EcalSrFlag::SRF_FULL);
    

    int dccZsFIRval = dccZsFIR(frame, firWeights_, firstFIRSample_, 0);

    if(highInterest){
      ++nEbHI_;
      // if(statusCode != 9) EBHighInterestZsFIR_->Fill( dccZsFIRval );
      EBHighInterestZsFIR_->Fill( dccZsFIRval );
    } else{//low interest
      ++nEbLI_;
      // if(statusCode != 9) EBLowInterestZsFIR_->Fill( dccZsFIRval );
      EBLowInterestZsFIR_->Fill( dccZsFIRval );
    }
    ++nPerDcc_[dccNum(id)-1];
    ++nPerRu_[ism-1][itt-1];
  }

}

void EBSelectiveReadoutTask::anaDigiInit(){
  nEb_ = 0;
  nEbLI_ = 0;
  nEbHI_ = 0;
  bzero(nPerDcc_, sizeof(nPerDcc_));
  bzero(nRuPerDcc_, sizeof(nRuPerDcc_));
  bzero(ebRuActive_, sizeof(ebRuActive_));

  for(int idcc=0; idcc<nECALDcc; idcc++) {
    for(int isc=0; isc<nDccChs; isc++) {
      nPerRu_[idcc][isc] = 0;
    }
  }

  for(int ietindex = 0; ietindex < 34; ietindex++ ) {
    for(int iptindex = 0; iptindex < 72; iptindex++ ) {
      nCryTower[iptindex][ietindex] = 0;
    }
  }

}

EcalTrigTowerDetId
EBSelectiveReadoutTask::readOutUnitOf(const EBDetId& xtalId) const{
  return xtalId.tower();
}

unsigned EBSelectiveReadoutTask::dccNum(const DetId& xtalId) const{
  int j;
  int k;

  if ( xtalId.det()!=DetId::Ecal ) {
    throw cms::Exception("EBSelectiveReadoutTask") << "Crystal does not belong to ECAL";
  }

  if(xtalId.subdetId()==EcalBarrel){
    EBDetId ebDetId(xtalId);
    j = iEta2cIndex(ebDetId.ieta());
    k = iPhi2cIndex(ebDetId.iphi());
  } else {
    throw cms::Exception("EBSelectiveReadoutTask") << "Not ECAL barrel.";
  }
  int iDcc0 = dccIndex(j,k);
  assert(iDcc0>=0 && iDcc0<nECALDcc);
  return iDcc0+1;
}

double EBSelectiveReadoutTask::getEbEventSize(double nReadXtals) const{
  double ruHeaderPayload = 0.;
  const int nEEDcc = 18;
  const int firstEbDcc0 = nEEDcc/2;
  for (int iDcc0 = firstEbDcc0; iDcc0 < firstEbDcc0 + nEBDcc; ++iDcc0 ) {
    ruHeaderPayload += nRuPerDcc_[iDcc0]*8.;
  }
  return getDccOverhead(EB)*nEBDcc +
         nReadXtals*bytesPerCrystal +
         ruHeaderPayload;
}

int EBSelectiveReadoutTask::dccPhiIndexOfRU(int i, int j) const {
  //iEta=i, iPhi=j
  //phi edge of a SM is 4 TT
  return j/4;
}

int EBSelectiveReadoutTask::dccIndex(int i, int j) const {
    //a SM is 85 crystal long:
    int iEtaSM = i/85;
    //a SM is 20 crystal wide:
    int iPhiSM = j/20;
    //DCC numbers start at 9 in the barrel and there 18 DCC/SM
    return 9+18*iEtaSM+iPhiSM;
}

//This implementation  assumes that int is coded on at least 28-bits,
//which in pratice should be always true.
int
EBSelectiveReadoutTask::dccZsFIR(const EcalDataFrame& frame,
                                 const std::vector<int>& firWeights,
                                 int firstFIRSample,
                                 bool* saturated){
  const int nFIRTaps = 6;
  //FIR filter weights:
  const std::vector<int>& w = firWeights;

  //accumulator used to compute weighted sum of samples
  int acc = 0;
  bool gain12saturated = false;
  const int gain12 = 0x01;

  int iWeight = 0;
  for(int i = -1; i < nFIRTaps - 1; ++i, ++iWeight){
    int iSample(i + firstFIRSample);
    if(iSample>=0 && iSample < frame.size()){
      EcalMGPASample sample(frame[iSample]);
      if(sample.gainId()!=gain12) gain12saturated = true;
      LogTrace("DccFir") << (iSample>=firstFIRSample?"+":"") << sample.adc()
                         << "*(" << w[iWeight] << ")";
      acc+=sample.adc()*w[iWeight];
    } else{
      edm::LogWarning("DccFir") << __FILE__ << ":" << __LINE__ <<
        ": Not enough samples in data frame or 'ecalDccZs1stSample' module "
        "parameter is not valid...";
    }
  }
  LogTrace("DccFir") << "\n";
  //discards the 8 LSBs
  //(shift operator cannot be used on negative numbers because
  // the result depends on compilator implementation)
  acc = (acc>=0)?(acc >> 8):-(-acc >> 8);
  //ZS passed if weighted sum acc above ZS threshold or if
  //one sample has a lower gain than gain 12 (that is gain 12 output
  //is saturated)

  LogTrace("DccFir") << "acc: " << acc << "\n"
                     << "saturated: " << (gain12saturated?"yes":"no") << "\n";

  if(saturated){
    *saturated = gain12saturated;
  }

  return gain12saturated?std::numeric_limits<int>::max():acc;
}

std::vector<int>
EBSelectiveReadoutTask::getFIRWeights(const std::vector<double>&
                                      normalizedWeights){
  const int nFIRTaps = 6;
  std::vector<int> firWeights(nFIRTaps, 0); //default weight: 0;
  const static int maxWeight = 0xEFF; //weights coded on 11+1 signed bits
  for(unsigned i=0; i < std::min((size_t)nFIRTaps,normalizedWeights.size()); ++i){
    firWeights[i] = lround(normalizedWeights[i] * (1<<10));
    if(std::abs(firWeights[i])>maxWeight){//overflow
      firWeights[i] = firWeights[i]<0?-maxWeight:maxWeight;
    }
  }
  return firWeights;
}

void
EBSelectiveReadoutTask::configFirWeights(std::vector<double> weightsForZsFIR){
  bool notNormalized  = false;
  bool notInt = false;
  for(unsigned i=0; i < weightsForZsFIR.size(); ++i){
    if(weightsForZsFIR[i] > 1.) notNormalized = true;
    if((int)weightsForZsFIR[i]!=weightsForZsFIR[i]) notInt = true;
  }
  if(notInt && notNormalized){
    throw cms::Exception("InvalidParameter")
      << "weigtsForZsFIR paramater values are not valid: they "
      << "must either be integer and uses the hardware representation "
      << "of the weights or less or equal than 1 and used the normalized "
      << "representation.";
  }
  edm::LogInfo log("DccFir");
  if(notNormalized){
    firWeights_ = std::vector<int>(weightsForZsFIR.size());
    for(unsigned i = 0; i< weightsForZsFIR.size(); ++i){
      firWeights_[i] = (int)weightsForZsFIR[i];
    }
  } else{
    firWeights_ = getFIRWeights(weightsForZsFIR);
  }

  log << "Input weights for FIR: ";
  for(unsigned i = 0; i < weightsForZsFIR.size(); ++i){
    log << weightsForZsFIR[i] << "\t";
  }

  double s2 = 0.;
  log << "\nActual FIR weights: ";
  for(unsigned i = 0; i < firWeights_.size(); ++i){
    log << firWeights_[i] << "\t";
    s2 += firWeights_[i]*firWeights_[i];
  }

  s2 = sqrt(s2);
  log << "\nNormalized FIR weights after hw representation rounding: ";
  for(unsigned i = 0; i < firWeights_.size(); ++i){
    log << firWeights_[i] / (double)(1<<10) << "\t";
  }

  log <<"\nFirst FIR sample: " << firstFIRSample_;
}

