/*
 * \file EESelectiveReadoutTask.cc
 *
 * $Date: 2008/12/03 12:55:50 $
 * $Revision: 1.20 $
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

  // histograms...
  EEDccEventSize_ = 0;

  EEReadoutUnitForcedBitMap_[0] = 0;
  EEFullReadoutSRFlagMap_[0] = 0;
  EEHighInterestTriggerTowerFlagMap_[0] = 0;
  EELowInterestTriggerTowerFlagMap_[0] = 0;
  EEEventSize_[0] = 0;
  EEHighInterestPayload_[0] = 0;
  EELowInterestPayload_[0] = 0;

  EEReadoutUnitForcedBitMap_[1] = 0;
  EEFullReadoutSRFlagMap_[1] = 0;
  EEHighInterestTriggerTowerFlagMap_[1] = 0;
  EELowInterestTriggerTowerFlagMap_[1] = 0;
  EEEventSize_[1] = 0;
  EEHighInterestPayload_[1] = 0;
  EELowInterestPayload_[1] = 0;

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

    sprintf(histo, "EESRT DCC event size");
    EEDccEventSize_ = dqmStore_->bookProfile(histo, histo, 18, 1, 19, 100, 0., 200., "s");
    for (int i = 0; i < 18; i++) {
      EEDccEventSize_->setBinLabel(i+1, Numbers::sEE(i+1).c_str(), 1);
    }

    sprintf(histo, "EESRT readout unit with SR forced EE -");
    EEReadoutUnitForcedBitMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT readout unit with SR forced EE +");
    EEReadoutUnitForcedBitMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_[1]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT full readout SR flags EE -");
    EEFullReadoutSRFlagMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT full readout SR flags EE +");
    EEFullReadoutSRFlagMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_[1]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT high interest TT Flags EE -");
    EEHighInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT high interest TT Flags EE +");
    EEHighInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT low interest TT Flags EE -");
    EELowInterestTriggerTowerFlagMap_[0] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT low interest TT Flags EE +");
    EELowInterestTriggerTowerFlagMap_[1] = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_[1]->setAxisTitle("jy", 2);

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

  }

}

void EESelectiveReadoutTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");

    if ( EEDccEventSize_ ) dqmStore_->removeElement( EEDccEventSize_->getName() );
    EEDccEventSize_ = 0;

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

  }

  init_ = false;

}

void EESelectiveReadoutTask::endJob(void){

  LogInfo("EESelectiveReadoutTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EESelectiveReadoutTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EESelectiveReadoutTask::endRun(const Run& r, const EventSetup& c) {

}

void EESelectiveReadoutTask::reset(void) {

  if ( EEDccEventSize_ ) EEDccEventSize_->Reset();

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

	EEDccEventSize_->Fill(ism, ((double)raw->FEDData(firstFedOnSide+iDcc).size())/kByte );

      }
    }

  } else {
    LogWarning("EESelectiveReadoutTask") << FEDRawDataCollection_ << " not available";
  }

  TH2F *h01[2];
  float integral01[2];
  for(int iside=0;iside<2;iside++) {
    h01[iside] = UtilsClient::getHisto<TH2F*>( EEFullReadoutSRFlagMap_[iside] );
    integral01[iside] = h01[iside]->GetEntries();
    if( integral01[iside] != 0 ) h01[iside]->Scale( integral01[iside] );
  }
  
  TH2F *h02[2];
  float integral02[2];
  for(int iside=0;iside<2;iside++) {
    h02[iside] = UtilsClient::getHisto<TH2F*>( EEReadoutUnitForcedBitMap_[iside] );
    integral02[iside] = h02[iside]->GetEntries();
    if( integral02[iside] != 0 ) h02[iside]->Scale( integral02[iside] );
  }

  // Selective Readout Flags
  Handle<EESrFlagCollection> eeSrFlags;
  if ( e.getByLabel(EESRFlagCollection_,eeSrFlags) ) {

    for ( EESrFlagCollection::const_iterator it = eeSrFlags->begin(); it != eeSrFlags->end(); ++it ) {

      int ix = it->id().ix();
      int iy = it->id().iy();

      int zside = it->id().zside();

      if ( zside < 0 ) ix = 101 - ix;

      float xix = ix-0.5;
      float xiy = iy-0.5;

      int flag = it->value() & ~EcalSrFlag::SRF_FORCED_MASK;

      if(flag == EcalSrFlag::SRF_FULL){
	if( zside < 0 ) {
	  EEFullReadoutSRFlagMap_[0]->Fill(xix,xiy);
	}
	else {
	  EEFullReadoutSRFlagMap_[1]->Fill(xix,xiy);
	}
      } else {
	if( zside < 0 ) {
	  EEFullReadoutSRFlagMap_[0]->Fill(-1,-1);
	}
	else {
	  EEFullReadoutSRFlagMap_[1]->Fill(-1,-1);
	}
      }

      if(it->value() & EcalSrFlag::SRF_FORCED_MASK){
	if( zside < 0 ) {
	  EEReadoutUnitForcedBitMap_[0]->Fill(xix,xiy);
	}
	else {
	  EEReadoutUnitForcedBitMap_[1]->Fill(xix,xiy);
	}
      } else {
	if( zside < 0 ) {
	  EEReadoutUnitForcedBitMap_[0]->Fill(-1,1);
	}
	else {
	  EEReadoutUnitForcedBitMap_[1]->Fill(-1,1);
	}
      }

    }
  } else {
    LogWarning("EESelectiveReadoutTask") << EESRFlagCollection_ << " not available";
  }

  for(int iside=0;iside<2;iside++) {
    if( integral01[iside] != 0 ) h01[iside]->Scale( 1.0/integral01[iside] );
    if( integral02[iside] != 0 ) h02[iside]->Scale( 1.0/integral02[iside] );  
  }

  TH2F *h03[2];
  float integral03[2];
  for(int iside=0;iside<2;iside++) {
    h03[iside] = UtilsClient::getHisto<TH2F*>( EELowInterestTriggerTowerFlagMap_[iside] );
    integral03[iside] = h03[iside]->GetEntries();
    if( integral03[iside] != 0 ) h03[iside]->Scale( integral03[iside] );
  }

  TH2F *h04[2];
  float integral04[2];
  for(int iside=0;iside<2;iside++) {
    h04[iside] = UtilsClient::getHisto<TH2F*>( EEHighInterestTriggerTowerFlagMap_[iside] );
    integral04[iside] = h04[iside]->GetEntries();
    if( integral04[iside] != 0 ) h04[iside]->Scale( integral04[iside] );
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

        float xix = ix-0.5;
        float xiy = iy-0.5;

        if ( (TPdigi->ttFlag() & 0x3) == 0 ) {
	  if ( ismt >= 1 && ismt <= 9 ) {
	    EELowInterestTriggerTowerFlagMap_[0]->Fill(xix,xiy);
	  }
	  else {
	    EELowInterestTriggerTowerFlagMap_[1]->Fill(xix,xiy);
	  }
        } else {
	  if ( ismt >= 1 && ismt <= 9 ) {
	    EELowInterestTriggerTowerFlagMap_[0]->Fill(-1,-1);
	  }
	  else {
	    EELowInterestTriggerTowerFlagMap_[1]->Fill(-1,-1);
	  }
	}

        if ( (TPdigi->ttFlag() & 0x3) == 3 ) {
          if ( ismt >= 1 && ismt <= 9 ) {
	    EEHighInterestTriggerTowerFlagMap_[0]->Fill(xix,xiy);
	  }
	  else {
	    EEHighInterestTriggerTowerFlagMap_[1]->Fill(xix,xiy);
	  }
        } else {
          if ( ismt >= 1 && ismt <= 9 ) {
	    EEHighInterestTriggerTowerFlagMap_[0]->Fill(-1,-1);
	  }
	  else {
	    EEHighInterestTriggerTowerFlagMap_[1]->Fill(-1,-1);
	  }
	}

      }

    }
  } else {
    LogWarning("EESelectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }

  for(int iside=0;iside<2;iside++) {
    if( integral03[iside] != 0 ) h03[iside]->Scale( 1.0/integral03[iside] );
    if( integral04[iside] != 0 ) h04[iside]->Scale( 1.0/integral04[iside] );
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

  } else {
    LogWarning("EESelectiveReadoutTask") << EEDigiCollection_ << " not available";
  }

}

void EESelectiveReadoutTask::anaDigi(const EEDataFrame& frame, const EESrFlagCollection& srFlagColl){

  EEDetId id = frame.id();
  EESrFlagCollection::const_iterator srf = srFlagColl.find(readOutUnitOf(id));

  if(srf == srFlagColl.end()){
    // LogWarning("EESelectiveReadoutTask") << "SR flag not found";
    return;
  }

  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
                       == EcalSrFlag::SRF_FULL);

  bool endcap = (id.subdetId()==EcalEndcap);

  if(endcap){
    int ism = Numbers::iSM( id );
    if ( ism >= 1 && ism <= 9 ) {
      ++nEe_[0];
      if(highInterest){
	++nEeHI_[0];
      } else{//low interest
	++nEeLI_[0];
      }
    } else {
      ++nEe_[1];
      if(highInterest){
	++nEeHI_[1];
      } else{//low interest
	++nEeLI_[1];
      }
    }

    int iX0 = iXY2cIndex(id.ix());
    int iY0 = iXY2cIndex(id.iy());
    int iZ0 = id.zside()>0?1:0;

    if(!eeRuActive_[iZ0][iX0/scEdge][iY0/scEdge]){
      ++nRuPerDcc_[dccNum(id)];
      eeRuActive_[iZ0][iX0/scEdge][iY0/scEdge] = true;
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
    throw cms::Exception("EESelectiveReadoutTask")
      <<"Not ECAL endcap.";
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
  return getDccOverhead(EE)*nEEDcc + nReadXtals*bytesPerCrystal
    + ruHeaderPayload;
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

