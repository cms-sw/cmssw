/*
 * \file EESelectiveReadoutTask.cc
 *
 * $Date: 2009/07/21 17:32:59 $
 * $Revision: 1.31 $
 * \author P. Gras
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EESelectiveReadoutTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EESelectiveReadoutTask::EESelectiveReadoutTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  // parameters...
  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EEUnsuppressedDigiCollection_ = ps.getParameter<edm::InputTag>("EEUsuppressedDigiCollection");
  EESRFlagCollection_ = ps.getParameter<edm::InputTag>("EESRFlagCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  firstFIRSample_ = ps.getParameter<int>("ecalDccZs1stSample");

  configFirWeights(ps.getParameter<vector<double> >("dccWeights"));

  // histograms...
  EEDccEventSize_ = 0;
  EEDccEventSizeMap_ = 0;

  EETowerSize_[0] = 0;
  EEReadoutUnitForcedBitMap_[0] = 0;
  EEFullReadoutSRFlagMap_[0] = 0;
  EEHighInterestTriggerTowerFlagMap_[0] = 0;
  EELowInterestTriggerTowerFlagMap_[0] = 0;
  EEEventSize_[0] = 0;
  EEHighInterestPayload_[0] = 0;
  EELowInterestPayload_[0] = 0;
  EEHighInterestZsFIR_[0] = 0;
  EELowInterestZsFIR_[0] = 0;

  EETowerSize_[1] = 0;
  EEReadoutUnitForcedBitMap_[1] = 0;
  EEFullReadoutSRFlagMap_[1] = 0;
  EEHighInterestTriggerTowerFlagMap_[1] = 0;
  EELowInterestTriggerTowerFlagMap_[1] = 0;
  EEEventSize_[1] = 0;
  EEHighInterestPayload_[1] = 0;
  EELowInterestPayload_[1] = 0;
  EEHighInterestZsFIR_[1] = 0;
  EELowInterestZsFIR_[1] = 0;

  // initialize variable binning for DCC size...
  float ZSthreshold = 0.608; // kBytes of 1 TT fully readout
  float zeroBinSize = ZSthreshold / 20.;
  for(int i=0; i<20; i++) ybins[i] = i*zeroBinSize;
  for(int i=20; i<133; i++) ybins[i] = ZSthreshold * (i-19);
  for(int i=0; i<=18; i++) xbins[i] = i+1;

}

EESelectiveReadoutTask::~EESelectiveReadoutTask() {

}

void EESelectiveReadoutTask::beginJob(const EventSetup& c) {

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");
    dqmStore_->rmdir(prefixME_ + "/EESelectiveReadoutTask");
  }

  Numbers::initGeometry(c, false);

}

void EESelectiveReadoutTask::setup(void) {

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");


    sprintf(histo, "EESRT tower event size EE -");
    EETowerSize_[0] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 20., 20, 0., 20., 100, 0., 200., "s");
    EETowerSize_[0]->setAxisTitle("jx", 1);
    EETowerSize_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT tower event size EE +");
    EETowerSize_[1] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 20., 20, 0., 20., 100, 0., 200., "s");
    EETowerSize_[1]->setAxisTitle("jx", 1);
    EETowerSize_[1]->setAxisTitle("jy", 2);
    
    sprintf(histo, "EESRT DCC event size");
    EEDccEventSize_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 200., "s");
    EEDccEventSize_->setAxisTitle("event size (kB)", 2);
    for (int i = 0; i < 18; i++) {
      EEDccEventSize_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    sprintf(histo, "EESRT event size vs DCC");
    EEDccEventSizeMap_ = dqmStore_->book2D(histo, histo, 18, xbins, 132, ybins);
    EEDccEventSizeMap_->setAxisTitle("event size (kB)", 2);
    for (int i = 0; i < 18; i++) {
      EEDccEventSizeMap_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    sprintf(histo, "EESRT readout unit with SR forced EE -");
    EEReadoutUnitForcedBitMap_[0] = dqmStore_->book2D(histo, histo, 20, 0., 20., 20, 0., 20.);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("jy", 2);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT readout unit with SR forced EE +");
    EEReadoutUnitForcedBitMap_[1] = dqmStore_->book2D(histo, histo, 20, 0., 20., 20, 0., 20.);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("jy", 2);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT full readout SR Flags EE -");
    EEFullReadoutSRFlagMap_[0] = dqmStore_->book2D(histo, histo, 20, 0., 20., 20, 0., 20.);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("jy", 2);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT full readout SR Flags EE +");
    EEFullReadoutSRFlagMap_[1] = dqmStore_->book2D(histo, histo, 20, 0., 20., 20, 0., 20.);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("jy", 2);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT high interest TT Flags EE -");
    EEHighInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT high interest TT Flags EE +");
    EEHighInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT low interest TT Flags EE -");
    EELowInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT low interest TT Flags EE +");
    EELowInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("rate", 3);

    sprintf(histo, "EESRT event size EE -");
    EEEventSize_[0] = dqmStore_->book1D(histo, histo, 100, 0, 200);
    EEEventSize_[0]->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT event size EE +");
    EEEventSize_[1] = dqmStore_->book1D(histo, histo, 100, 0, 200);
    EEEventSize_[1]->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT high interest payload EE -");
    EEHighInterestPayload_[0] =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EEHighInterestPayload_[0]->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT high interest payload EE +");
    EEHighInterestPayload_[1] =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EEHighInterestPayload_[1]->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT low interest payload EE -");
    EELowInterestPayload_[0] =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EELowInterestPayload_[0]->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT low interest payload EE +");
    EELowInterestPayload_[1] =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EELowInterestPayload_[1]->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT high interest ZS filter output EE -");
    EEHighInterestZsFIR_[0] = dqmStore_->book1D(histo, histo, 60, -30, 30);
    EEHighInterestZsFIR_[0]->setAxisTitle("ADC counts*4",1);

    sprintf(histo, "EESRT high interest ZS filter output EE +");
    EEHighInterestZsFIR_[1] = dqmStore_->book1D(histo, histo, 60, -30, 30);
    EEHighInterestZsFIR_[1]->setAxisTitle("ADC counts*4",1);

    sprintf(histo, "EESRT low interest ZS filter output EE -");
    EELowInterestZsFIR_[0] = dqmStore_->book1D(histo, histo, 60, -30, 30);
    EELowInterestZsFIR_[0]->setAxisTitle("ADC counts*4",1);

    sprintf(histo, "EESRT low interest ZS filter output EE +");
    EELowInterestZsFIR_[1] = dqmStore_->book1D(histo, histo, 60, -30, 30);
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

    if ( EEHighInterestTriggerTowerFlagMap_[0] ) dqmStore_->removeElement( EEHighInterestTriggerTowerFlagMap_[0]->getName() );
    EEHighInterestTriggerTowerFlagMap_[0] = 0;

    if ( EEHighInterestTriggerTowerFlagMap_[1] ) dqmStore_->removeElement( EEHighInterestTriggerTowerFlagMap_[1]->getName() );
    EEHighInterestTriggerTowerFlagMap_[1] = 0;

    if ( EELowInterestTriggerTowerFlagMap_[0] ) dqmStore_->removeElement( EELowInterestTriggerTowerFlagMap_[0]->getName() );
    EELowInterestTriggerTowerFlagMap_[0] = 0;

    if ( EELowInterestTriggerTowerFlagMap_[1] ) dqmStore_->removeElement( EELowInterestTriggerTowerFlagMap_[1]->getName() );
    EELowInterestTriggerTowerFlagMap_[1] = 0;

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

  LogInfo("EESelectiveReadoutTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EESelectiveReadoutTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for(int ix = 0; ix < 20; ix++ ) {
    for(int iy = 0; iy < 20; iy++ ) {
      for(int iz = 0; iz < 2; iz++) {
        nEvtFullReadout[ix][iy][iz] = 0;
        nEvtRUForced[ix][iy][iz] = 0;
        nEvtAnyReadout[ix][iy][iz] = 0;
        nEvtHighInterest[ix][iy][iz] = 0;
        nEvtLowInterest[ix][iy][iz] = 0;
        nEvtAnyInterest[ix][iy][iz] = 0;
      }
    }
  }
  
}

void EESelectiveReadoutTask::endRun(const Run& r, const EventSetup& c) {

}

void EESelectiveReadoutTask::reset(void) {

  if ( EETowerSize_[0] ) EETowerSize_[0]->Reset();
  if ( EETowerSize_[1] ) EETowerSize_[1]->Reset();
  
  if ( EEDccEventSize_ ) EEDccEventSize_->Reset();

  if ( EEDccEventSizeMap_ ) EEDccEventSizeMap_->Reset();

  if ( EEReadoutUnitForcedBitMap_[0] ) EEReadoutUnitForcedBitMap_[0]->Reset();
  if ( EEReadoutUnitForcedBitMap_[1] ) EEReadoutUnitForcedBitMap_[1]->Reset();

  if ( EEFullReadoutSRFlagMap_[0] ) EEFullReadoutSRFlagMap_[0]->Reset();
  if ( EEFullReadoutSRFlagMap_[1] ) EEFullReadoutSRFlagMap_[1]->Reset();

  if ( EEHighInterestTriggerTowerFlagMap_[0] ) EEHighInterestTriggerTowerFlagMap_[0]->Reset();
  if ( EEHighInterestTriggerTowerFlagMap_[1] ) EEHighInterestTriggerTowerFlagMap_[1]->Reset();

  if ( EELowInterestTriggerTowerFlagMap_[0] ) EELowInterestTriggerTowerFlagMap_[0]->Reset();
  if ( EELowInterestTriggerTowerFlagMap_[1] ) EELowInterestTriggerTowerFlagMap_[1]->Reset();

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

void EESelectiveReadoutTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<FEDRawDataCollection> raw;
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
    LogWarning("EESelectiveReadoutTask") << FEDRawDataCollection_ << " not available";
  }

  // Selective Readout Flags
  Handle<EESrFlagCollection> eeSrFlags;
  if ( e.getByLabel(EESRFlagCollection_,eeSrFlags) ) {

    for ( EESrFlagCollection::const_iterator it = eeSrFlags->begin(); it != eeSrFlags->end(); ++it ) {

      int ix = it->id().ix();
      int iy = it->id().iy();

      int zside = it->id().zside();

      int iz = ( zside < 0 ) ? 0 : 1;

      if ( zside < 0 ) ix = 21 - ix;

      nEvtAnyReadout[ix-1][iy-1][iz]++;

      int flag = it->value() & ~EcalSrFlag::SRF_FORCED_MASK;

      if(flag == EcalSrFlag::SRF_FULL) nEvtFullReadout[ix-1][iy-1][iz]++;

      if(it->value() & EcalSrFlag::SRF_FORCED_MASK) nEvtRUForced[ix-1][iy-1][iz]++;

    }
  } else {
    LogWarning("EESelectiveReadoutTask") << EESRFlagCollection_ << " not available";
  }

  for(int iz = 0; iz < 2; iz++) {
    for(int ix = 0; ix < 20; ix++ ) {
      for(int iy = 0; iy < 20; iy++ ) {

        if( nEvtAnyReadout[ix][iy][iz] ) {

          float xix = ix;
          if ( iz == 0 ) xix = 19 - xix;
          xix += 0.5;

          float xiy = iy+0.5;

          float fraction = float(nEvtFullReadout[ix][iy][iz] / nEvtAnyReadout[ix][iy][iz]);
          float error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));

          TH2F *h2d = EEFullReadoutSRFlagMap_[iz]->getTH2F();

          int binx=0, biny=0;
          
          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }
          
          EEFullReadoutSRFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EEFullReadoutSRFlagMap_[iz]->setBinError(binx, biny, error);


          fraction = float(nEvtRUForced[ix][iy][iz] / nEvtAnyReadout[ix][iy][iz]);
          error = sqrt(fraction*(1-fraction)/float(nEvtAnyReadout[ix][iy][iz]));

          h2d = EEReadoutUnitForcedBitMap_[iz]->getTH2F();

          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }
          
          EEReadoutUnitForcedBitMap_[iz]->setBinContent(binx, biny, fraction);
          EEReadoutUnitForcedBitMap_[iz]->setBinError(binx, biny, error);

        }

      }
    }
  }

  Handle<EcalTrigPrimDigiCollection> TPCollection;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, TPCollection) ) {

    // Trigger Primitives
    EcalTrigPrimDigiCollection::const_iterator TPdigi;
    for ( TPdigi = TPCollection->begin(); TPdigi != TPCollection->end(); ++TPdigi ) {

      if ( Numbers::subDet( TPdigi->id() ) != EcalEndcap ) continue;

      int ismt = Numbers::iSM( TPdigi->id() );
      
      vector<DetId> crystals = Numbers::crystals( TPdigi->id() );
      
      for ( unsigned int i=0; i<crystals.size(); i++ ) {
        
        EEDetId id = crystals[i];
        
        int ix = id.ix();
        int iy = id.iy();
        
        if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;
        
        int zside = TPdigi->id().zside();
        int iz = ( zside < 0 ) ? 0 : 1;
        
        nEvtAnyInterest[ix-1][iy-1][iz]++;
        
        if ( (TPdigi->ttFlag() & 0x3) == 0 ) nEvtLowInterest[ix-1][iy-1][iz]++;
        
        if ( (TPdigi->ttFlag() & 0x3) == 3 ) nEvtHighInterest[ix-1][iy-1][iz]++;
        
      }

    }
  } else {
    LogWarning("EESelectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }

  for(int iz = 0; iz < 2; iz++) {
    for(int ix = 0; ix < 100; ix++ ) {
      for(int iy = 0; iy < 100; iy++ ) {

        if( nEvtAnyInterest[ix][iy][iz] ) {

          float xix = ix;
          if ( iz == 0 ) xix = 99 - xix;
          xix += 0.5;

          float xiy = iy+0.5;

          float fraction = float(nEvtHighInterest[ix][iy][iz] / nEvtAnyInterest[ix][iy][iz]);
          float error = sqrt(fraction*(1-fraction)/float(nEvtAnyInterest[ix][iy][iz]));

          TH2F *h2d = EEHighInterestTriggerTowerFlagMap_[iz]->getTH2F();

          int binx=0, biny=0;
          
          if( h2d ) {
            binx = h2d->GetXaxis()->FindBin(xix);
            biny = h2d->GetYaxis()->FindBin(xiy);
          }
          
          EEHighInterestTriggerTowerFlagMap_[iz]->setBinContent(binx, biny, fraction);
          EEHighInterestTriggerTowerFlagMap_[iz]->setBinError(binx, biny, error);


          fraction = float(nEvtLowInterest[ix][iy][iz] / nEvtAnyInterest[ix][iy][iz]);
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

  if (!eeSrFlags.isValid()) return;

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

  Handle<EEDigiCollection> eeDigis;
  if ( e.getByLabel(EEDigiCollection_ , eeDigis) ) {

    anaDigiInit();

    for (unsigned int digis=0; digis<eeDigis->size(); ++digis) {
      EEDataFrame eedf = (*eeDigis)[digis];
      anaDigi(eedf, *eeSrFlags);
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

          double towerSize =  nCryTower[ix][iy][iz] * bytesPerCrystal;

          float xix = ix;

          if ( iz == 0 ) xix = 19 - xix;

          xix += 0.5;
          float xiy = iy+0.5;
     
          EETowerSize_[iz]->Fill(xix, xiy, towerSize);

        }
      }
    }

  } else {
    LogWarning("EESelectiveReadoutTask") << EEDigiCollection_ << " not available";
  }

}

void EESelectiveReadoutTask::anaDigi(const EEDataFrame& frame, const EESrFlagCollection& srFlagColl){

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
    
  }

  EESrFlagCollection::const_iterator srf = srFlagColl.find(readOutUnitOf(id));

  if(srf == srFlagColl.end()){
    return;
  }

  int ttix = srf->id().ix();
  int ttiy = srf->id().iy();

  int zside = srf->id().zside();

  int ttiz = ( zside < 0 ) ? 0 : 1;

  nCryTower[ttix-1][ttiy-1][ttiz]++;

  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
                       == EcalSrFlag::SRF_FULL);

  if(endcap) {

    int dccZsFIRval = dccZsFIR(frame, firWeights_, firstFIRSample_, 0); 

    if ( ism >= 1 && ism <= 9 ) {
      if(highInterest) {
	++nEeHI_[0];
        EEHighInterestZsFIR_[0]->Fill( dccZsFIRval );
      } else{ //low interest
	++nEeLI_[0];
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
  }

  ++nPerDcc_[dccNum(id)-1];
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
  
  for(int iz = 0; iz<2; iz++) {
    for(int ix = 0; ix < 20; ix++ ) {
      for(int iy = 0; iy < 20; iy++ ) {
        nCryTower[ix][iy][iz] = 0;
      }
    }
  }

}

EcalScDetId
EESelectiveReadoutTask::readOutUnitOf(const EEDetId& xtalId) const{
  const int scEdge = 5;
  return EcalScDetId((xtalId.ix()-1)/scEdge+1,
                     (xtalId.iy()-1)/scEdge+1,
                     xtalId.zside());
}

unsigned EESelectiveReadoutTask::dccNum(const DetId& xtalId) const{
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
  const vector<int>& w = firWeights;
  
  //accumulator used to compute weighted sum of samples
  int acc = 0;
  bool gain12saturated = false;
  const int gain12 = 0x01; 
  const int lastFIRSample = firstFIRSample + nFIRTaps - 1;
  //LogDebug("DccFir") << "DCC FIR operation: ";
  int iWeight = 0;
  for(int iSample=firstFIRSample-1;
      iSample<lastFIRSample; ++iSample, ++iWeight){
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
  
  return gain12saturated?numeric_limits<int>::max():acc;
}

std::vector<int>
EESelectiveReadoutTask::getFIRWeights(const std::vector<double>&
                                      normalizedWeights){
  const int nFIRTaps = 6;
  vector<int> firWeights(nFIRTaps, 0); //default weight: 0;
  const static int maxWeight = 0xEFF; //weights coded on 11+1 signed bits
  for(unsigned i=0; i < min((size_t)nFIRTaps,normalizedWeights.size()); ++i){ 
    firWeights[i] = lround(normalizedWeights[i] * (1<<10));
    if(abs(firWeights[i])>maxWeight){//overflow
      firWeights[i] = firWeights[i]<0?-maxWeight:maxWeight;
    }
  }
  return firWeights;
}

void
EESelectiveReadoutTask::configFirWeights(vector<double> weightsForZsFIR){
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
  LogInfo log("DccFir");
  if(notNormalized){
    firWeights_ = vector<int>(weightsForZsFIR.size());
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

