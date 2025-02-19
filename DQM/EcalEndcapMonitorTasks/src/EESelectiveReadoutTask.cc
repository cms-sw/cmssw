/*
 * \file EESelectiveReadoutTask.cc
 *
 * $Date: 2012/04/27 13:46:16 $
 * $Revision: 1.66 $
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

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EESelectiveReadoutTask.h"

#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"


EESelectiveReadoutTask::EESelectiveReadoutTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  // parameters...
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EEUnsuppressedDigiCollection_ = ps.getParameter<edm::InputTag>("EEUsuppressedDigiCollection");
  EESRFlagCollection_ = ps.getParameter<edm::InputTag>("EESRFlagCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  firstFIRSample_ = ps.getParameter<int>("ecalDccZs1stSample");

  useCondDb_ = ps.getParameter<bool>("configFromCondDB");
  if(!useCondDb_) configFirWeights(ps.getParameter<std::vector<double> >("dccWeights"));

  // histograms...
  EEDccEventSize_ = 0;
  EEDccEventSizeMap_ = 0;

  for(int i=0; i<2; i++) {
    EETowerSize_[i] = 0;
    EETTFMismatch_[i] = 0;
    EEReadoutUnitForcedBitMap_[i] = 0;
    EEFullReadoutSRFlagMap_[i] = 0;
    EEFullReadoutSRFlagCount_[i] = 0;
    EEZeroSuppression1SRFlagMap_[i] = 0;  
    EEHighInterestTriggerTowerFlagMap_[i] = 0;
    EEMediumInterestTriggerTowerFlagMap_[i] = 0;
    EELowInterestTriggerTowerFlagMap_[i] = 0;
    EETTFlags_[i] = 0;
    EECompleteZSMap_[i] = 0;
    EECompleteZSCount_[i] = 0;
    EEDroppedFRMap_[i] = 0;
    EEDroppedFRCount_[i] = 0;
    EEEventSize_[i] = 0;
    EEHighInterestPayload_[i] = 0;
    EELowInterestPayload_[i] = 0;
    EEHighInterestZsFIR_[i] = 0;
    EELowInterestZsFIR_[i] = 0;
  }

  // initialize variable binning for DCC size...
  float ZSthreshold = 0.608; // kBytes of 1 TT fully readout
  float zeroBinSize = ZSthreshold / 20.;
  for(int i=0; i<20; i++) ybins[i] = i*zeroBinSize;
  for(int i=20; i<133; i++) ybins[i] = ZSthreshold * (i-19);
  for(int i=0; i<=18; i++) xbins[i] = i+1;

}

EESelectiveReadoutTask::~EESelectiveReadoutTask() {

}

void EESelectiveReadoutTask::beginJob(void) {

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");
    dqmStore_->rmdir(prefixME_ + "/EESelectiveReadoutTask");
  }

}

void EESelectiveReadoutTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

  for(int ix = 0; ix < 20; ix++ ) {
    for(int iy = 0; iy < 20; iy++ ) {
      for(int iz = 0; iz < 2; iz++) {
        nEvtFullReadout[ix][iy][iz] = 0;
        nEvtZS1Readout[ix][iy][iz] = 0;
        nEvtZSReadout[ix][iy][iz] = 0;
        nEvtCompleteReadoutIfZS[ix][iy][iz] = 0;
        nEvtDroppedReadoutIfFR[ix][iy][iz] = 0;
        nEvtRUForced[ix][iy][iz] = 0;
        nEvtAnyReadout[ix][iy][iz] = 0;
      }
    }
  }
  for(int ix = 0; ix < 100; ix++ ) {
    for(int iy = 0; iy < 100; iy++ ) {
      for(int iz = 0; iz < 2; iz++) {
        nEvtHighInterest[ix][iy][iz] = 0;
        nEvtMediumInterest[ix][iy][iz] = 0;
        nEvtLowInterest[ix][iy][iz] = 0;
        nEvtAnyInterest[ix][iy][iz] = 0;
      }
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
    else edm::LogWarning("EESelectiveReadoutTask") << "DCC weight set is not exactly 1.";

    configFirWeights(wsFromDB);
  }

}

void EESelectiveReadoutTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EESelectiveReadoutTask::reset(void) {

  if ( EETowerSize_[0] ) EETowerSize_[0]->Reset();
  if ( EETowerSize_[1] ) EETowerSize_[1]->Reset();

  if ( EETTFMismatch_[0] ) EETTFMismatch_[0]->Reset();
  if ( EETTFMismatch_[1] ) EETTFMismatch_[1]->Reset();

  if ( EEDccEventSize_ ) EEDccEventSize_->Reset();

  if ( EEDccEventSizeMap_ ) EEDccEventSizeMap_->Reset();

  if ( EEReadoutUnitForcedBitMap_[0] ) EEReadoutUnitForcedBitMap_[0]->Reset();
  if ( EEReadoutUnitForcedBitMap_[1] ) EEReadoutUnitForcedBitMap_[1]->Reset();

  if ( EEFullReadoutSRFlagMap_[0] ) EEFullReadoutSRFlagMap_[0]->Reset();
  if ( EEFullReadoutSRFlagMap_[1] ) EEFullReadoutSRFlagMap_[1]->Reset();

  if ( EEFullReadoutSRFlagCount_[0] ) EEFullReadoutSRFlagCount_[0]->Reset();
  if ( EEFullReadoutSRFlagCount_[1] ) EEFullReadoutSRFlagCount_[1]->Reset();
  
  if ( EEZeroSuppression1SRFlagMap_[0] ) EEZeroSuppression1SRFlagMap_[0]->Reset();
  if ( EEZeroSuppression1SRFlagMap_[1] ) EEZeroSuppression1SRFlagMap_[1]->Reset();

  if ( EEHighInterestTriggerTowerFlagMap_[0] ) EEHighInterestTriggerTowerFlagMap_[0]->Reset();
  if ( EEHighInterestTriggerTowerFlagMap_[1] ) EEHighInterestTriggerTowerFlagMap_[1]->Reset();

  if ( EEMediumInterestTriggerTowerFlagMap_[0] ) EEMediumInterestTriggerTowerFlagMap_[0]->Reset();
  if ( EEMediumInterestTriggerTowerFlagMap_[1] ) EEMediumInterestTriggerTowerFlagMap_[1]->Reset();

  if ( EELowInterestTriggerTowerFlagMap_[0] ) EELowInterestTriggerTowerFlagMap_[0]->Reset();
  if ( EELowInterestTriggerTowerFlagMap_[1] ) EELowInterestTriggerTowerFlagMap_[1]->Reset();

  if ( EETTFlags_[0] ) EETTFlags_[0]->Reset();
  if ( EETTFlags_[1] ) EETTFlags_[1]->Reset();

  if ( EECompleteZSMap_[0] ) EECompleteZSMap_[0]->Reset();
  if ( EECompleteZSMap_[1] ) EECompleteZSMap_[1]->Reset();
  
  if ( EECompleteZSCount_[0] ) EECompleteZSCount_[0]->Reset();
  if ( EECompleteZSCount_[1] ) EECompleteZSCount_[1]->Reset();

  if ( EEDroppedFRMap_[0] ) EEDroppedFRMap_[0]->Reset();
  if ( EEDroppedFRMap_[1] ) EEDroppedFRMap_[1]->Reset();

  if ( EEDroppedFRCount_[0] ) EEDroppedFRCount_[0]->Reset();
  if ( EEDroppedFRCount_[1] ) EEDroppedFRCount_[1]->Reset();

  if ( EEEventSize_[0] ) EEEventSize_[0]->Reset();
  if ( EEEventSize_[1] ) EEEventSize_[1]->Reset();

  if ( EEHighInterestPayload_[0] ) EEHighInterestPayload_[0]->Reset();
  if ( EEHighInterestPayload_[1] ) EEHighInterestPayload_[1]->Reset();

  if ( EELowInterestPayload_[0] ) EELowInterestPayload_[0]->Reset();
  if ( EELowInterestPayload_[1] ) EELowInterestPayload_[1]->Reset();

  if ( EEHighInterestZsFIR_[0] ) EEHighInterestZsFIR_[0]->Reset();
  if ( EEHighInterestZsFIR_[1] ) EEHighInterestZsFIR_[1]->Reset();

  if ( EELowInterestZsFIR_[0] ) EELowInterestZsFIR_[0]->Reset();
  if ( EELowInterestZsFIR_[1] ) EELowInterestZsFIR_[1]->Reset();

}

void EESelectiveReadoutTask::setup(void) {

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");

    name = "EESRT tower event size EE -";
    EETowerSize_[0] = dqmStore_->bookProfile2D(name, name, 20, 0., 20., 20, 0., 20., 100, 0., 200., "s");
    EETowerSize_[0]->setAxisTitle("jx", 1);
    EETowerSize_[0]->setAxisTitle("jy", 2);

    name = "EESRT tower event size EE +";
    EETowerSize_[1] = dqmStore_->bookProfile2D(name, name, 20, 0., 20., 20, 0., 20., 100, 0., 200., "s");
    EETowerSize_[1]->setAxisTitle("jx", 1);
    EETowerSize_[1]->setAxisTitle("jy", 2);

    name = "EESRT TT flag mismatch EE -";
    EETTFMismatch_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EETTFMismatch_[0]->setAxisTitle("jx", 1);
    EETTFMismatch_[0]->setAxisTitle("jy", 2);

    name = "EESRT TT flag mismatch EE +";
    EETTFMismatch_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EETTFMismatch_[1]->setAxisTitle("jx", 1);
    EETTFMismatch_[1]->setAxisTitle("jy", 2);

    name = "EESRT DCC event size";
    EEDccEventSize_ = dqmStore_->bookProfile(name, name, 18, 1, 19, 100, 0., 200., "s");
    EEDccEventSize_->setAxisTitle("event size (kB)", 2);
    for (int i = 0; i < 18; i++) {
      EEDccEventSize_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EESRT event size vs DCC";
    EEDccEventSizeMap_ = dqmStore_->book2D(name, name, 18, xbins, 132, ybins);
    EEDccEventSizeMap_->setAxisTitle("event size (kB)", 2);
    for (int i = 0; i < 18; i++) {
      EEDccEventSizeMap_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    name = "EESRT readout unit with SR forced EE -";
    EEReadoutUnitForcedBitMap_[0] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("jy", 2);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT readout unit with SR forced EE +";
    EEReadoutUnitForcedBitMap_[1] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("jy", 2);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT full readout SR Flags EE -";
    EEFullReadoutSRFlagMap_[0] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("jy", 2);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT full readout SR Flags EE +";
    EEFullReadoutSRFlagMap_[1] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("jy", 2);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT full readout SR Flags Number EE -";
    EEFullReadoutSRFlagCount_[0] = dqmStore_->book1D(name, name, 200, 0., 200.);
    EEFullReadoutSRFlagCount_[0]->setAxisTitle("Readout Units number", 1);

    name = "EESRT full readout SR Flags Number EE +";
    EEFullReadoutSRFlagCount_[1] = dqmStore_->book1D(name, name, 200, 0., 200.);
    EEFullReadoutSRFlagCount_[1]->setAxisTitle("Fully readout RU number", 1);

    name = "EESRT zero suppression 1 SR Flags EE -";
    EEZeroSuppression1SRFlagMap_[0] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEZeroSuppression1SRFlagMap_[0]->setAxisTitle("jx", 1);
    EEZeroSuppression1SRFlagMap_[0]->setAxisTitle("jy", 2);
    EEZeroSuppression1SRFlagMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT zero suppression 1 SR Flags EE +";
    EEZeroSuppression1SRFlagMap_[1] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEZeroSuppression1SRFlagMap_[1]->setAxisTitle("jx", 1);
    EEZeroSuppression1SRFlagMap_[1]->setAxisTitle("jy", 2);
    EEZeroSuppression1SRFlagMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT high interest TT Flags EE -";
    EEHighInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT high interest TT Flags EE +";
    EEHighInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT medium interest TT Flags EE -";
    EEMediumInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EEMediumInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EEMediumInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);
    EEMediumInterestTriggerTowerFlagMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT medium interest TT Flags EE +";
    EEMediumInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EEMediumInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EEMediumInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);
    EEMediumInterestTriggerTowerFlagMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT low interest TT Flags EE -";
    EELowInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT low interest TT Flags EE +";
    EELowInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(name, name, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT TT Flags EE -";
    EETTFlags_[0] = dqmStore_->book1D(name, name, 8, 0., 8.);
    EETTFlags_[0]->setAxisTitle("TT Flag value", 1);

    name = "EESRT TT Flags EE +";
    EETTFlags_[1] = dqmStore_->book1D(name, name, 8, 0., 8.);
    EETTFlags_[1]->setAxisTitle("TT Flag value", 1);

    name = "EESRT ZS Flagged Fully Readout EE -";
    EECompleteZSMap_[0] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EECompleteZSMap_[0]->setAxisTitle("jphi", 1);
    EECompleteZSMap_[0]->setAxisTitle("jeta", 2);
    EECompleteZSMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT ZS Flagged Fully Readout EE +";
    EECompleteZSMap_[1] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EECompleteZSMap_[1]->setAxisTitle("jphi", 1);
    EECompleteZSMap_[1]->setAxisTitle("jeta", 2);
    EECompleteZSMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT ZS Flagged Fully Readout Number EE -";
    EECompleteZSCount_[0] = dqmStore_->book1D(name, name, 20, 0., 20.);
    EECompleteZSCount_[0]->setAxisTitle("Readout Units number", 1);

    name = "EESRT ZS Flagged Fully Readout Number EE +";
    EECompleteZSCount_[1] = dqmStore_->book1D(name, name, 20, 0., 20.);
    EECompleteZSCount_[1]->setAxisTitle("Readout Units number", 1);

    name = "EESRT FR Flagged Dropped Readout EE -";
    EEDroppedFRMap_[0] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEDroppedFRMap_[0]->setAxisTitle("jphi", 1);
    EEDroppedFRMap_[0]->setAxisTitle("jeta", 2);
    EEDroppedFRMap_[0]->setAxisTitle("rate", 3);

    name = "EESRT FR Flagged Dropped Readout EE +";
    EEDroppedFRMap_[1] = dqmStore_->book2D(name, name, 20, 0., 20., 20, 0., 20.);
    EEDroppedFRMap_[1]->setAxisTitle("jphi", 1);
    EEDroppedFRMap_[1]->setAxisTitle("jeta", 2);
    EEDroppedFRMap_[1]->setAxisTitle("rate", 3);

    name = "EESRT FR Flagged Dropped Readout Number EE -";
    EEDroppedFRCount_[0] = dqmStore_->book1D(name, name, 20, 0., 20.);
    EEDroppedFRCount_[0]->setAxisTitle("Readout Units number", 1);

    name = "EESRT FR Flagged Dropped Readout Number EE +";
    EEDroppedFRCount_[1] = dqmStore_->book1D(name, name, 20, 0., 20.);
    EEDroppedFRCount_[1]->setAxisTitle("Readout Units number", 1);

    name = "EESRT event size EE -";
    EEEventSize_[0] = dqmStore_->book1D(name, name, 100, 0, 200);
    EEEventSize_[0]->setAxisTitle("event size (kB)",1);

    name = "EESRT event size EE +";
    EEEventSize_[1] = dqmStore_->book1D(name, name, 100, 0, 200);
    EEEventSize_[1]->setAxisTitle("event size (kB)",1);

    name = "EESRT high interest payload EE -";
    EEHighInterestPayload_[0] =  dqmStore_->book1D(name, name, 100, 0, 200);
    EEHighInterestPayload_[0]->setAxisTitle("event size (kB)",1);

    name = "EESRT high interest payload EE +";
    EEHighInterestPayload_[1] =  dqmStore_->book1D(name, name, 100, 0, 200);
    EEHighInterestPayload_[1]->setAxisTitle("event size (kB)",1);

    name = "EESRT low interest payload EE -";
    EELowInterestPayload_[0] =  dqmStore_->book1D(name, name, 100, 0, 200);
    EELowInterestPayload_[0]->setAxisTitle("event size (kB)",1);

    name = "EESRT low interest payload EE +";
    EELowInterestPayload_[1] =  dqmStore_->book1D(name, name, 100, 0, 200);
    EELowInterestPayload_[1]->setAxisTitle("event size (kB)",1);

    name = "EESRT high interest ZS filter output EE -";
    EEHighInterestZsFIR_[0] = dqmStore_->book1D(name, name, 60, -30, 30);
    EEHighInterestZsFIR_[0]->setAxisTitle("ADC counts*4",1);

    name = "EESRT high interest ZS filter output EE +";
    EEHighInterestZsFIR_[1] = dqmStore_->book1D(name, name, 60, -30, 30);
    EEHighInterestZsFIR_[1]->setAxisTitle("ADC counts*4",1);

    name = "EESRT low interest ZS filter output EE -";
    EELowInterestZsFIR_[0] = dqmStore_->book1D(name, name, 60, -30, 30);
    EELowInterestZsFIR_[0]->setAxisTitle("ADC counts*4",1);

    name = "EESRT low interest ZS filter output EE +";
    EELowInterestZsFIR_[1] = dqmStore_->book1D(name, name, 60, -30, 30);
    EELowInterestZsFIR_[1]->setAxisTitle("ADC counts*4",1);

  }

}

void EESelectiveReadoutTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");

    if ( EETowerSize_[0] ) dqmStore_->removeElement( EETowerSize_[0]->getName() );
    EETowerSize_[0] = 0;

    if ( EETowerSize_[1] ) dqmStore_->removeElement( EETowerSize_[1]->getName() );
    EETowerSize_[1] = 0;

    if ( EETTFMismatch_[0] ) dqmStore_->removeElement( EETTFMismatch_[0]->getName() );
    EETTFMismatch_[0] = 0;

    if ( EETTFMismatch_[1] ) dqmStore_->removeElement( EETTFMismatch_[1]->getName() );
    EETTFMismatch_[1] = 0;

    if ( EEDccEventSize_ ) dqmStore_->removeElement( EEDccEventSize_->getName() );
    EEDccEventSize_ = 0;

    if ( EEDccEventSizeMap_ ) dqmStore_->removeElement( EEDccEventSizeMap_->getName() );
    EEDccEventSizeMap_ = 0;

    if ( EEReadoutUnitForcedBitMap_[0] ) dqmStore_->removeElement( EEReadoutUnitForcedBitMap_[0]->getName() );
    EEReadoutUnitForcedBitMap_[0] = 0;

    if ( EEReadoutUnitForcedBitMap_[1] ) dqmStore_->removeElement( EEReadoutUnitForcedBitMap_[1]->getName() );
    EEReadoutUnitForcedBitMap_[1] = 0;

    if ( EEFullReadoutSRFlagMap_[0] ) dqmStore_->removeElement( EEFullReadoutSRFlagMap_[0]->getName() );
    EEFullReadoutSRFlagMap_[0] = 0;

    if ( EEFullReadoutSRFlagMap_[1] ) dqmStore_->removeElement( EEFullReadoutSRFlagMap_[1]->getName() );
    EEFullReadoutSRFlagMap_[1] = 0;

    if ( EEFullReadoutSRFlagCount_[0] ) dqmStore_->removeElement( EEFullReadoutSRFlagCount_[0]->getName() );
    EEFullReadoutSRFlagCount_[0] = 0;

    if ( EEFullReadoutSRFlagCount_[1] ) dqmStore_->removeElement( EEFullReadoutSRFlagCount_[1]->getName() );
    EEFullReadoutSRFlagCount_[1] = 0;

    if ( EEZeroSuppression1SRFlagMap_[0] ) dqmStore_->removeElement( EEZeroSuppression1SRFlagMap_[0]->getName() );
    EEZeroSuppression1SRFlagMap_[0] = 0;

    if ( EEZeroSuppression1SRFlagMap_[1] ) dqmStore_->removeElement( EEZeroSuppression1SRFlagMap_[1]->getName() );
    EEZeroSuppression1SRFlagMap_[1] = 0;

    if ( EEHighInterestTriggerTowerFlagMap_[0] ) dqmStore_->removeElement( EEHighInterestTriggerTowerFlagMap_[0]->getName() );
    EEHighInterestTriggerTowerFlagMap_[0] = 0;

    if ( EEHighInterestTriggerTowerFlagMap_[1] ) dqmStore_->removeElement( EEHighInterestTriggerTowerFlagMap_[1]->getName() );
    EEHighInterestTriggerTowerFlagMap_[1] = 0;

    if ( EELowInterestTriggerTowerFlagMap_[0] ) dqmStore_->removeElement( EELowInterestTriggerTowerFlagMap_[0]->getName() );
    EELowInterestTriggerTowerFlagMap_[0] = 0;

    if ( EELowInterestTriggerTowerFlagMap_[1] ) dqmStore_->removeElement( EELowInterestTriggerTowerFlagMap_[1]->getName() );
    EELowInterestTriggerTowerFlagMap_[1] = 0;

    if ( EETTFlags_[0] ) dqmStore_->removeElement( EETTFlags_[0]->getName() );
    EETTFlags_[0] = 0;

    if ( EETTFlags_[1] ) dqmStore_->removeElement( EETTFlags_[1]->getName() );
    EETTFlags_[1] = 0;

    if ( EECompleteZSMap_[0] ) dqmStore_->removeElement( EECompleteZSMap_[0]->getName() );
    EECompleteZSMap_[0] = 0;

    if ( EECompleteZSMap_[1] ) dqmStore_->removeElement( EECompleteZSMap_[1]->getName() );
    EECompleteZSMap_[1] = 0;

    if ( EECompleteZSCount_[0] ) dqmStore_->removeElement( EECompleteZSCount_[0]->getName() );
    EECompleteZSCount_[0] = 0;

    if ( EECompleteZSCount_[1] ) dqmStore_->removeElement( EECompleteZSCount_[1]->getName() );
    EECompleteZSCount_[1] = 0;

    if ( EEDroppedFRMap_[0] ) dqmStore_->removeElement( EEDroppedFRMap_[0]->getName() );
    EEDroppedFRMap_[0] = 0;

    if ( EEDroppedFRMap_[1] ) dqmStore_->removeElement( EEDroppedFRMap_[1]->getName() );
    EEDroppedFRMap_[1] = 0;

    if ( EEDroppedFRCount_[0] ) dqmStore_->removeElement( EEDroppedFRCount_[0]->getName() );
    EEDroppedFRCount_[0] = 0;

    if ( EEDroppedFRCount_[1] ) dqmStore_->removeElement( EEDroppedFRCount_[1]->getName() );
    EEDroppedFRCount_[1] = 0;

    if ( EEEventSize_[0] ) dqmStore_->removeElement( EEEventSize_[0]->getName() );
    EEEventSize_[0] = 0;

    if ( EEEventSize_[1] ) dqmStore_->removeElement( EEEventSize_[1]->getName() );
    EEEventSize_[1] = 0;

    if ( EEHighInterestPayload_[0] ) dqmStore_->removeElement( EEHighInterestPayload_[0]->getName() );
    EEHighInterestPayload_[0] = 0;

    if ( EEHighInterestPayload_[1] ) dqmStore_->removeElement( EEHighInterestPayload_[1]->getName() );
    EEHighInterestPayload_[1] = 0;

    if ( EELowInterestPayload_[0] ) dqmStore_->removeElement( EELowInterestPayload_[0]->getName() );
    EELowInterestPayload_[0] = 0;

    if ( EELowInterestPayload_[1] ) dqmStore_->removeElement( EELowInterestPayload_[1]->getName() );
    EELowInterestPayload_[1] = 0;

    if ( EEHighInterestZsFIR_[0] ) dqmStore_->removeElement( EEHighInterestZsFIR_[0]->getName() );
    EEHighInterestZsFIR_[0] = 0;

    if ( EEHighInterestZsFIR_[1] ) dqmStore_->removeElement( EEHighInterestZsFIR_[1]->getName() );
    EEHighInterestZsFIR_[1] = 0;

    if ( EELowInterestZsFIR_[0] ) dqmStore_->removeElement( EELowInterestZsFIR_[0]->getName() );
    EELowInterestZsFIR_[0] = 0;

    if ( EELowInterestZsFIR_[1] ) dqmStore_->removeElement( EELowInterestZsFIR_[1]->getName() );
    EELowInterestZsFIR_[1] = 0;

  }

  init_ = false;

}

void EESelectiveReadoutTask::endJob(void){

  edm::LogInfo("EESelectiveReadoutTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EESelectiveReadoutTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  edm::Handle<FEDRawDataCollection> raw;
  if ( e.getByLabel(FEDRawDataCollection_, raw) ) {

    int EEFirstFED[2];
    EEFirstFED[0] = 601; // EE-
    EEFirstFED[1] = 646; // EE+
    for(int zside=0; zside<2; zside++) {

      int firstFedOnSide=EEFirstFED[zside];

      for ( int iDcc = 0; iDcc < 9; ++iDcc ) {

	int ism = 0;
	if ( zside == 0 ) ism = iDcc+1;
	else ism = 10+iDcc;

	EEDccEventSize_->Fill( ism, ((double)raw->FEDData(firstFedOnSide+iDcc).size())/kByte );
	EEDccEventSizeMap_->Fill( ism, ((double)raw->FEDData(firstFedOnSide+iDcc).size())/kByte );

      }
    }

  } else {
    edm::LogWarning("EESelectiveReadoutTask") << FEDRawDataCollection_ << " not available";
  }

  // Selective Readout Flags
  int nFRO[2], nCompleteZS[2], nDroppedFRO[2];
  nFRO[0] = nFRO[1] = 0;
  nCompleteZS[0] = nCompleteZS[1] = 0;
  nDroppedFRO[0] = nDroppedFRO[1] = 0;
  edm::Handle<EESrFlagCollection> eeSrFlags;
  if ( e.getByLabel(EESRFlagCollection_,eeSrFlags) ) {

    // Data Volume
    double aLowInterest[2];
    double aHighInterest[2];
    double aAnyInterest[2];

    aLowInterest[0]=0;
    aHighInterest[0]=0;
    aAnyInterest[0]=0;
    aLowInterest[1]=0;
    aHighInterest[1]=0;
    aAnyInterest[1]=0;

    edm::Handle<EEDigiCollection> eeDigis;
    if ( e.getByLabel(EEDigiCollection_ , eeDigis) ) {

      anaDigiInit();

      // channel status
      edm::ESHandle<EcalChannelStatus> pChannelStatus;
      c.get<EcalChannelStatusRcd>().get(pChannelStatus);
      const EcalChannelStatus* chStatus = pChannelStatus.product();  

      for (unsigned int digis=0; digis<eeDigis->size(); ++digis) {
        EEDataFrame eedf = (*eeDigis)[digis];
        EEDetId id = eedf.id();
        EcalChannelStatusMap::const_iterator chit;
        chit = chStatus->getMap().find(id.rawId());
        uint16_t statusCode = 0;
        if( chit != chStatus->getMap().end() ) {
          EcalChannelStatusCode ch_code = (*chit);
          statusCode = ch_code.getStatusCode();
        }
        anaDigi(eedf, *eeSrFlags, statusCode);
      }

      //low interest channels:
      aLowInterest[0] = nEeLI_[0]*bytesPerCrystal/kByte;
      EELowInterestPayload_[0]->Fill(aLowInterest[0]);
      aLowInterest[1] = nEeLI_[1]*bytesPerCrystal/kByte;
      EELowInterestPayload_[1]->Fill(aLowInterest[1]);

      //low interest channels:
      aHighInterest[0] = nEeHI_[0]*bytesPerCrystal/kByte;
      EEHighInterestPayload_[0]->Fill(aHighInterest[0]);
      aHighInterest[1] = nEeHI_[1]*bytesPerCrystal/kByte;
      EEHighInterestPayload_[1]->Fill(aHighInterest[1]);

      //any-interest channels:
      aAnyInterest[0] = getEeEventSize(nEe_[0])/kByte;
      EEEventSize_[0]->Fill(aAnyInterest[0]);
      aAnyInterest[1] = getEeEventSize(nEe_[1])/kByte;
      EEEventSize_[1]->Fill(aAnyInterest[1]);

      //event size by tower:
      for(int ix = 0; ix < 20; ix++ ) {
        for(int iy = 0; iy < 20; iy++ ) {
          for(int iz = 0; iz < 2; iz++) {

            double towerSize =  nCrySC[ix][iy][iz] * bytesPerCrystal;

            float xix = ix;
            if ( iz == 0 ) xix = 19 - xix;
            xix += 0.5;

            float xiy = iy+0.5;

            EETowerSize_[iz]->Fill(xix, xiy, towerSize);

          }
        }
      }

    } else {
      edm::LogWarning("EESelectiveReadoutTask") << EEDigiCollection_ << " not available";
    }

    // initialize dcchs_ to mask disabled towers
    std::map< int, std::vector<short> > towersStatus;
    edm::Handle<EcalRawDataCollection> dcchs;

    if( e.getByLabel(FEDRawDataCollection_, dcchs) ) {
      for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {
        if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;
        int ism = Numbers::iSM( *dcchItr, EcalEndcap );
        towersStatus.insert(std::make_pair(ism, dcchItr->getFEStatus()));
      }
    }

    for ( EESrFlagCollection::const_iterator it = eeSrFlags->begin(); it != eeSrFlags->end(); ++it ) {

      EcalScDetId id = it->id();

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int ix = id.ix();
      int iy = id.iy();
      int iDcc = dccNumOfRU(id);
      int ism = Numbers::iSM( id );
      int isc = Numbers::iSC( id );

      int zside = id.zside();

      int iz = ( zside < 0 ) ? 0 : 1;

      if ( zside < 0 ) ix = 21 - ix;

      nEvtAnyReadout[ix-1][iy-1][iz]++;

      int flag = it->value() & ~EcalSrFlag::SRF_FORCED_MASK;

      int status=0;
      if( towersStatus[ism].size() > 0 ) status = (towersStatus[ism])[isc];

      if(flag == EcalSrFlag::SRF_FULL) {
        nEvtFullReadout[ix-1][iy-1][iz]++;
        nFRO[iz]++;
        if(nPerRu_[iDcc-1][isc] == 0) {
          if(status != 1) nEvtDroppedReadoutIfFR[ix-1][iy-1][iz]++;
          nDroppedFRO[iz]++;
        }
      }

      if(flag == EcalSrFlag::SRF_ZS1) nEvtZS1Readout[ix-1][iy-1][iz]++;

      if(it->value() & EcalSrFlag::SRF_FORCED_MASK) nEvtRUForced[ix-1][iy-1][iz]++;

      if(flag == EcalSrFlag::SRF_ZS1 || flag == EcalSrFlag::SRF_ZS2) {
        nEvtZSReadout[ix-1][iy-1][iz]++;
        if(nPerRu_[iDcc-1][isc] == getCrystalCount(iDcc,isc)) {
          if(status != 1) nEvtCompleteReadoutIfZS[ix-1][iy-1][iz]++;
          nCompleteZS[iz]++;
        }
      }

    }
  } else {
    edm::LogWarning("EESelectiveReadoutTask") << EESRFlagCollection_ << " not available";
  }

  for(int ix = 0; ix < 20; ix++ ) {
    for(int iy = 0; iy < 20; iy++ ) {
      for(int iz = 0; iz < 2; iz++) {

        if( nEvtAnyReadout[ix][iy][iz] ) {

          float xix = ix;
          if ( iz == 0 ) xix = 19 - xix;
          xix += 0.5;

          float xiy = iy+0.5;

          float fraction = float(nEvtFullReadout[ix][iy][iz]) / float(nEvtAnyReadout[ix][iy][iz]);
          float error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));

          TH2F *h2d = EEFullReadoutSRFlagMap_[iz]->getTH2F();

          int binx=0, biny=0;

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }

          EEFullReadoutSRFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EEFullReadoutSRFlagMap_[iz]->setBinError(binx, biny, error);


          fraction = float(nEvtZS1Readout[ix][iy][iz]) / float(nEvtAnyReadout[ix][iy][iz]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));

          h2d = EEZeroSuppression1SRFlagMap_[iz]->getTH2F();

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }

          EEZeroSuppression1SRFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EEZeroSuppression1SRFlagMap_[iz]->setBinError(binx, biny, error);


          fraction = float(nEvtRUForced[ix][iy][iz]) / float(nEvtAnyReadout[ix][iy][iz]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));

          h2d = EEReadoutUnitForcedBitMap_[iz]->getTH2F();

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }

          EEReadoutUnitForcedBitMap_[iz]->setBinContent(binx, biny, fraction);
          EEReadoutUnitForcedBitMap_[iz]->setBinError(binx, biny, error);

          if( nEvtZSReadout[ix][iy][iz] ) {
            fraction = float(nEvtCompleteReadoutIfZS[ix][iy][iz]) / float(nEvtZSReadout[ix][iy][iz]);
            error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));
            
            h2d = EECompleteZSMap_[iz]->getTH2F();
            
            if( h2d ) {
              binx = h2d->GetXaxis()->FindBin(xix);
              biny = h2d->GetYaxis()->FindBin(xiy);
            }
            
            EECompleteZSMap_[iz]->setBinContent(binx, biny, fraction);
            EECompleteZSMap_[iz]->setBinError(binx, biny, error);
          }

          if( nEvtFullReadout[ix][iy][iz] ) {
            fraction = float(nEvtDroppedReadoutIfFR[ix][iy][iz]) / float(nEvtFullReadout[ix][iy][iz]);
            error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));
            
            h2d = EEDroppedFRMap_[iz]->getTH2F();
            
            if( h2d ) {
              binx = h2d->GetXaxis()->FindBin(xix);
              biny = h2d->GetYaxis()->FindBin(xiy);
            }

            EEDroppedFRMap_[iz]->setBinContent(binx, biny, fraction);
            EEDroppedFRMap_[iz]->setBinError(binx, biny, error);
          }

        }

      }
    }
  }

  for(int iz = 0; iz < 2; iz++) {
    EEFullReadoutSRFlagCount_[iz]->Fill( nFRO[iz] );
    EECompleteZSCount_[iz]->Fill( nCompleteZS[iz] );
    EEDroppedFRCount_[iz]->Fill( nDroppedFRO[iz] );
  }

  edm::Handle<EcalTrigPrimDigiCollection> TPCollection;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, TPCollection) ) {

    // Trigger Primitives
    EcalTrigPrimDigiCollection::const_iterator TPdigi;
    for ( TPdigi = TPCollection->begin(); TPdigi != TPCollection->end(); ++TPdigi ) {

      if ( Numbers::subDet( TPdigi->id() ) != EcalEndcap ) continue;

      int ismt = Numbers::iSM( TPdigi->id() );
      int zside = TPdigi->id().zside();
      int iz = ( zside < 0 ) ? 0 : 1;

      EETTFlags_[iz]->Fill( TPdigi->ttFlag() );

      std::vector<DetId>* crystals = Numbers::crystals( TPdigi->id() );

      for ( unsigned int i=0; i<crystals->size(); i++ ) {

        EEDetId id = (*crystals)[i];

        int ix = id.ix();
        int iy = id.iy();
        int ism = Numbers::iSM( id );
        int itcc = Numbers::iTCC( ism, EcalEndcap, ix, iy );
        int itt = Numbers::iTT( ism, EcalEndcap, ix, iy );

        if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

        nEvtAnyInterest[ix-1][iy-1][iz]++;

        if ( (TPdigi->ttFlag() & 0x3) == 0 ) nEvtLowInterest[ix-1][iy-1][iz]++;

        if ( (TPdigi->ttFlag() & 0x3) == 1 ) nEvtMediumInterest[ix-1][iy-1][iz]++;

        if ( (TPdigi->ttFlag() & 0x3) == 3 ) nEvtHighInterest[ix-1][iy-1][iz]++;

        float xix = ix-0.5;
        if ( iz == 0 ) xix = 100 - xix;
        float xiy = iy-0.5;

        if ( ((TPdigi->ttFlag() & 0x3) == 1 || (TPdigi->ttFlag() & 0x3) == 3)
             && nCryTT[itcc-1][itt-1] != (int)crystals->size() ) EETTFMismatch_[iz]->Fill(xix, xiy);

      }

    }
  } else {
    edm::LogWarning("EESelectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }

  for(int ix = 0; ix < 100; ix++ ) {
    for(int iy = 0; iy < 100; iy++ ) {
      for(int iz = 0; iz < 2; iz++) {

        if( nEvtAnyInterest[ix][iy][iz] ) {

          float xix = ix;
          if ( iz == 0 ) xix = 99 - xix;
          xix += 0.5;

          float xiy = iy+0.5;

          float fraction = float(nEvtHighInterest[ix][iy][iz]) / float(nEvtAnyInterest[ix][iy][iz]);
          float error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[ix][iy][iz]));

          TH2F *h2d = EEHighInterestTriggerTowerFlagMap_[iz]->getTH2F();

          int binx=0, biny=0;

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }

          EEHighInterestTriggerTowerFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EEHighInterestTriggerTowerFlagMap_[iz]->setBinError(binx, biny, error);


          fraction = float(nEvtMediumInterest[ix][iy][iz]) / float(nEvtAnyInterest[ix][iy][iz]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[ix][iy][iz]));

          h2d = EEMediumInterestTriggerTowerFlagMap_[iz]->getTH2F();

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }

          EEMediumInterestTriggerTowerFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EEMediumInterestTriggerTowerFlagMap_[iz]->setBinError(binx, biny, error);


          fraction = float(nEvtLowInterest[ix][iy][iz]) / float(nEvtAnyInterest[ix][iy][iz]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[ix][iy][iz]));

          h2d = EELowInterestTriggerTowerFlagMap_[iz]->getTH2F();

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }

          EELowInterestTriggerTowerFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EELowInterestTriggerTowerFlagMap_[iz]->setBinError(binx, biny, error);

        }

      }
    }
  }

}

void EESelectiveReadoutTask::anaDigi(const EEDataFrame& frame, const EESrFlagCollection& srFlagColl, uint16_t statusCode){
  
  EEDetId id = frame.id();
  int ism = Numbers::iSM( id );

  bool endcap = (id.subdetId()==EcalEndcap);

  if(endcap) {
    if ( ism >= 1 && ism <= 9 ) {
      ++nEe_[0];
    } else {
      ++nEe_[1];
    }

    int ix = id.ix();
    int iy = id.iy();

    int iX0 = iXY2cIndex(ix);
    int iY0 = iXY2cIndex(iy);
    int iZ0 = id.zside()>0?1:0;

    if(!eeRuActive_[iZ0][iX0/scEdge][iY0/scEdge]){
      ++nRuPerDcc_[dccNum(id)];
      eeRuActive_[iZ0][iX0/scEdge][iY0/scEdge] = true;
    }

    EESrFlagCollection::const_iterator srf = srFlagColl.find(readOutUnitOf(id));

    if(srf == srFlagColl.end()){
      return;
    }

    int ttix = srf->id().ix();
    int ttiy = srf->id().iy();

    int zside = srf->id().zside();

    int ttiz = ( zside < 0 ) ? 0 : 1;

    nCrySC[ttix-1][ttiy-1][ttiz]++;

    int itcc = Numbers::iTCC( ism, EcalEndcap, ix, iy );
    int itt = Numbers::iTT( ism, EcalEndcap, ix, iy );
    nCryTT[itcc-1][itt-1]++;

    bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
                         == EcalSrFlag::SRF_FULL);

    int dccZsFIRval = dccZsFIR(frame, firWeights_, firstFIRSample_, 0);

    if ( ism >= 1 && ism <= 9 ) {
      if(highInterest) {
	++nEeHI_[0];
        // if(statusCode != 9) EEHighInterestZsFIR_[0]->Fill( dccZsFIRval );
        EEHighInterestZsFIR_[0]->Fill( dccZsFIRval );
      } else{ //low interest
	++nEeLI_[0];
        // if(statusCode != 9) EELowInterestZsFIR_[0]->Fill( dccZsFIRval );
        EELowInterestZsFIR_[0]->Fill( dccZsFIRval );
      }
    } else {
      if(highInterest) {
	++nEeHI_[1];
        EEHighInterestZsFIR_[1]->Fill( dccZsFIRval );
      } else{ //low interest
	++nEeLI_[1];
        EELowInterestZsFIR_[1]->Fill( dccZsFIRval );
      }
    }
    int isc = Numbers::iSC( ism, EcalEndcap, ix, iy );
    ++nPerDcc_[dccNum(id)-1];
    ++nPerRu_[dccNum(id)-1][isc];
  }
  
}

void EESelectiveReadoutTask::anaDigiInit(){
  nEe_[0] = 0;
  nEeLI_[0] = 0;
  nEeHI_[0] = 0;
  nEe_[1] = 0;
  nEeLI_[1] = 0;
  nEeHI_[1] = 0;
  bzero(nPerDcc_, sizeof(nPerDcc_));
  bzero(nRuPerDcc_, sizeof(nRuPerDcc_));
  bzero(eeRuActive_, sizeof(eeRuActive_));

  for(int idcc=0; idcc<nECALDcc; idcc++) {
    for(int isc=0; isc<nDccChs; isc++) {
      nPerRu_[idcc][isc] = 0;
    }
  }

  for(int iz = 0; iz<2; iz++) {
    for(int ix = 0; ix < 20; ix++ ) {
      for(int iy = 0; iy < 20; iy++ ) {
        nCrySC[ix][iy][iz] = 0;
      }
    }
  }

  for (int itcc = 0; itcc < 108; itcc++) {
    for (int itt = 0; itt < 41; itt++) nCryTT[itcc][itt] = 0;
  }

}

const EcalScDetId
EESelectiveReadoutTask::readOutUnitOf(const EEDetId& xtalId) const {
  if (xtalId.ix() > 40 && xtalId.ix() < 61 &&
      xtalId.iy() > 40 && xtalId.iy() < 61) {
    // crystal belongs to an inner partial supercrystal
    return Numbers::getEcalScDetId(xtalId);
  } else {
    return EcalScDetId((xtalId.ix()-1)/5+1, (xtalId.iy()-1)/5+1, xtalId.zside());
  }
}

unsigned EESelectiveReadoutTask::dccNum(const DetId& xtalId) const {
  int j;
  int k;

  if ( xtalId.det()!=DetId::Ecal ) {
    throw cms::Exception("EESelectiveReadoutTask") << "Crystal does not belong to ECAL";
  }

  int iDet = 0;

  if(xtalId.subdetId()==EcalEndcap){
    EEDetId eeDetId(xtalId);
    j = iXY2cIndex(eeDetId.ix());
    k = iXY2cIndex(eeDetId.iy());
    int zside = eeDetId.zside();
    if ( zside < 0 ) iDet = 0;
    else iDet = 2;
  } else {
    throw cms::Exception("EESelectiveReadoutTask") << "Not ECAL endcap.";
  }
  int iDcc0 = dccIndex(iDet,j,k);
  assert(iDcc0>=0 && iDcc0<nECALDcc);
  return iDcc0+1;
}

unsigned EESelectiveReadoutTask::dccNumOfRU(const EcalScDetId& scId) const {
  int j;
  int k;

  if ( scId.det()!=DetId::Ecal ) {
    throw cms::Exception("EESelectiveReadoutTask") << "SuperCrystal does not belong to ECAL";
  }

  int iDet = 0;

  if(scId.subdetId()==EcalEndcap){
    j = scId.ix()-1;
    k = scId.iy()-1;
    int zside = scId.zside();
    if ( zside < 0 ) iDet = 0;
    else iDet = 2;
  } else {
    throw cms::Exception("EESelectiveReadoutTask") << "Not ECAL endcap.";
  }
  int iDcc0 = 0;
  int iPhi = dccPhiIndexOfRU(j,k);
  if(iPhi<0) iDcc0 = -1;
  else iDcc0 = iPhi+iDet/2*45;
  assert(iDcc0>=0 && iDcc0<nECALDcc);
  return iDcc0+1;
}

double EESelectiveReadoutTask::getEeEventSize(double nReadXtals) const {
  double ruHeaderPayload = 0.;
  const int firstEbDcc0 = nEEDcc/2;
  for ( int iDcc0 = 0; iDcc0 < nECALDcc; ++iDcc0 ) {
    //skip barrel:
    if(iDcc0 == firstEbDcc0) iDcc0 += nEBDcc;
      ruHeaderPayload += nRuPerDcc_[iDcc0]*8.;
  }
  return getDccOverhead(EE)*nEEDcc +
         nReadXtals*bytesPerCrystal +
         ruHeaderPayload;
}

int EESelectiveReadoutTask::dccPhiIndexOfRU(int i, int j) const {
  char flag=endcapDccMap[i+j*20];
  return (flag==' ')?-1:(flag-'0');
}

int EESelectiveReadoutTask::dccIndex(int iDet, int i, int j) const {
  int iPhi = dccPhiIndex(i, j);
  if(iPhi<0) return -1;
  //34 DCCs in barrel and 8 in EE-=>in EE+ DCC numbering starts at 45,
  //iDet/2 is 0 for EE- and 1 for EE+:
  return iPhi+iDet/2*45;
}

//This implementation  assumes that int is coded on at least 28-bits,
//which in pratice should be always true.
int
EESelectiveReadoutTask::dccZsFIR(const EcalDataFrame& frame,
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
    int iSample(firstFIRSample + i);
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
EESelectiveReadoutTask::getFIRWeights(const std::vector<double>&
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
EESelectiveReadoutTask::configFirWeights(std::vector<double> weightsForZsFIR){
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

int EESelectiveReadoutTask::getCrystalCount(int iDcc, int iDccCh) {
  if(iDcc<1 || iDcc>54) {
    // invalid DCC
    return 0;
  } else if (10 <= iDcc && iDcc <= 45) {
    // EB
    return 25;
  } else {
    // EE
    int iDccPhi;
    if(iDcc < 10) {
      iDccPhi = iDcc;
    } else {
      iDccPhi = iDcc - 45;
    }
    switch(iDccPhi*100+iDccCh){
      case 110:
      case 232:
      case 312:
      case 412:
      case 532:
      case 610:
      case 830:
      case 806:
        //inner partials at 12, 3, and 9 o'clock
        return 20;
      case 134:
      case 634:
      case 827:
      case 803:
      return 10;
      case 330:
      case 430:
        return 20;
      case 203:
      case 503:
      case 721:
      case 921:
        return 21;
      default:
        return 25;
    }
  }
}
