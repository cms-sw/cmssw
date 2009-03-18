/*
 * \file EBSelectiveReadoutTask.cc
 *
 * $Date: 2009/03/18 08:23:48 $
 * $Revision: 1.27 $
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

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <DQM/EcalCommon/interface/UtilsClient.h>
#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBSelectiveReadoutTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBSelectiveReadoutTask::EBSelectiveReadoutTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  // parameters...
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EBUnsuppressedDigiCollection_ = ps.getParameter<edm::InputTag>("EBUsuppressedDigiCollection");
  EBSRFlagCollection_ = ps.getParameter<edm::InputTag>("EBSRFlagCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");

  // histograms...
  EBDccEventSize_ = 0;
  EBReadoutUnitForcedBitMap_ = 0;
  EBFullReadoutSRFlagMap_ = 0;
  EBHighInterestTriggerTowerFlagMap_ = 0;
  EBLowInterestTriggerTowerFlagMap_ = 0;
  EBEventSize_ = 0;
  EBHighInterestPayload_ = 0;
  EBLowInterestPayload_ = 0;

}

EBSelectiveReadoutTask::~EBSelectiveReadoutTask() {

}

void EBSelectiveReadoutTask::beginJob(const EventSetup& c) {

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");
    dqmStore_->rmdir(prefixME_ + "/EBSelectiveReadoutTask");
  }

  Numbers::initGeometry(c, false);

}

void EBSelectiveReadoutTask::setup(void) {

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");

    sprintf(histo, "EBSRT DCC event size");
    EBDccEventSize_ = dqmStore_->bookProfile(histo, histo, 36, 1, 37, 100, 0., 200., "s");
    for (int i = 0; i < 36; i++) {
      EBDccEventSize_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    sprintf(histo, "EBSRT readout unit with SR forced");
    EBReadoutUnitForcedBitMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EBReadoutUnitForcedBitMap_->setAxisTitle("jphi", 1);
    EBReadoutUnitForcedBitMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT full readout SR flags");
    EBFullReadoutSRFlagMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EBFullReadoutSRFlagMap_->setAxisTitle("jphi", 1);
    EBFullReadoutSRFlagMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT high interest TT Flags");
    EBHighInterestTriggerTowerFlagMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EBHighInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EBHighInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT low interest TT Flags");
    EBLowInterestTriggerTowerFlagMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EBLowInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EBLowInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT event size");
    EBEventSize_ = dqmStore_->book1D(histo, histo, 100, 0, 200);
    EBEventSize_->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EBSRT high interest payload");
    EBHighInterestPayload_ =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EBHighInterestPayload_->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EBSRT low interest payload");
    EBLowInterestPayload_ =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EBLowInterestPayload_->setAxisTitle("event size (kB)",1);

  }

}

void EBSelectiveReadoutTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");

    if ( EBDccEventSize_ ) dqmStore_->removeElement( EBDccEventSize_->getName() );
    EBDccEventSize_ = 0;

    if ( EBReadoutUnitForcedBitMap_ ) dqmStore_->removeElement( EBReadoutUnitForcedBitMap_->getName() );
    EBReadoutUnitForcedBitMap_ = 0;

    if ( EBFullReadoutSRFlagMap_ ) dqmStore_->removeElement( EBFullReadoutSRFlagMap_->getName() );
    EBFullReadoutSRFlagMap_ = 0;

    if ( EBHighInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EBHighInterestTriggerTowerFlagMap_->getName() );
    EBHighInterestTriggerTowerFlagMap_ = 0;

    if ( EBLowInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EBLowInterestTriggerTowerFlagMap_->getName() );
    EBLowInterestTriggerTowerFlagMap_ = 0;

    if ( EBEventSize_ ) dqmStore_->removeElement( EBEventSize_->getName() );
    EBEventSize_ = 0;

    if ( EBHighInterestPayload_ ) dqmStore_->removeElement( EBHighInterestPayload_->getName() );
    EBHighInterestPayload_ = 0;

    if ( EBLowInterestPayload_ ) dqmStore_->removeElement( EBLowInterestPayload_->getName() );
    EBLowInterestPayload_ = 0;

  }

  init_ = false;

}

void EBSelectiveReadoutTask::endJob(void){

  LogInfo("EBSelectiveReadoutTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBSelectiveReadoutTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EBSelectiveReadoutTask::endRun(const Run& r, const EventSetup& c) {

}

void EBSelectiveReadoutTask::reset(void) {

  if ( EBDccEventSize_ ) EBDccEventSize_->Reset();

  if ( EBReadoutUnitForcedBitMap_ ) EBReadoutUnitForcedBitMap_->Reset();

  if ( EBFullReadoutSRFlagMap_ ) EBFullReadoutSRFlagMap_->Reset();

  if ( EBHighInterestTriggerTowerFlagMap_ ) EBHighInterestTriggerTowerFlagMap_->Reset();

  if ( EBLowInterestTriggerTowerFlagMap_ ) EBLowInterestTriggerTowerFlagMap_->Reset();

  if ( EBEventSize_ ) EBEventSize_->Reset();

  if ( EBHighInterestPayload_ ) EBHighInterestPayload_->Reset();

  if ( EBLowInterestPayload_ ) EBLowInterestPayload_->Reset();

}

void EBSelectiveReadoutTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<FEDRawDataCollection> raw;
  if ( e.getByLabel(FEDRawDataCollection_, raw) ) {

    for ( int iDcc = 0; iDcc < nEBDcc; ++iDcc ) {

      EBDccEventSize_->Fill(iDcc+1, ((double)raw->FEDData(610+iDcc).size())/kByte );

    }

  } else {
    LogWarning("EBSelectiveReadoutTask") << FEDRawDataCollection_ << " not available";
  }
  
  TH2F *h01 = UtilsClient::getHisto<TH2F*>( EBFullReadoutSRFlagMap_ );
  float integral01 = h01->GetEntries();
  if( integral01 != 0 ) h01->Scale( integral01 );

  TH2F *h02 = UtilsClient::getHisto<TH2F*>( EBReadoutUnitForcedBitMap_ );
  float integral02 = h02->GetEntries();
  if( integral02 != 0 ) h02->Scale( integral02 );

  // Selective Readout Flags
  Handle<EBSrFlagCollection> ebSrFlags;
  if ( e.getByLabel(EBSRFlagCollection_,ebSrFlags) ) {

    for ( EBSrFlagCollection::const_iterator it = ebSrFlags->begin(); it != ebSrFlags->end(); ++it ) {

      int iet = it->id().ieta();
      // phi_tower: change the range from global to SM-local
      // phi==0 is in the middle of a SM
      int ipt = it->id().iphi() + 2;
      if ( ipt > 72 ) ipt = ipt - 72;

      float xiet = (iet>0) ? iet-0.5 : iet+0.5 ;
      float xipt = ipt-0.5;

      int flag = it->value() & ~EcalSrFlag::SRF_FORCED_MASK;

      if(flag == EcalSrFlag::SRF_FULL){
        EBFullReadoutSRFlagMap_->Fill(xipt,xiet);
      } else {
	EBFullReadoutSRFlagMap_->Fill(-1,-18);
      }

      if(it->value() & EcalSrFlag::SRF_FORCED_MASK){
        EBReadoutUnitForcedBitMap_->Fill(xipt,xiet);
      } else {
	EBReadoutUnitForcedBitMap_->Fill(-1,-18);
      }
      
    }
  } else {
    LogWarning("EBSelectiveReadoutTask") << EBSRFlagCollection_ << " not available";
  }

  integral01 = h01->GetEntries();
  if( integral01 != 0 ) h01->Scale( 1.0/integral01 );
  integral02 = h02->GetEntries();
  if( integral02 != 0 ) h02->Scale( 1.0/integral02 );

  TH2F *h03 = UtilsClient::getHisto<TH2F*>( EBLowInterestTriggerTowerFlagMap_ );
  float integral03 = h03->GetEntries();
  if( integral03 != 0 ) h03->Scale( integral03 );

  TH2F *h04 = UtilsClient::getHisto<TH2F*>( EBHighInterestTriggerTowerFlagMap_ );
  float integral04 = h04->GetEntries();
  if( integral04 != 0 ) h04->Scale( integral04 );

  Handle<EcalTrigPrimDigiCollection> TPCollection;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, TPCollection) ) {

    // Trigger Primitives
    EcalTrigPrimDigiCollection::const_iterator TPdigi;
    for (TPdigi = TPCollection->begin(); TPdigi != TPCollection->end(); ++TPdigi ) {

      if ( Numbers::subDet( TPdigi->id() ) != EcalBarrel ) continue;

      int iet = TPdigi->id().ieta();
      // phi_tower: change the range from global to SM-local
      // phi==0 is in the middle of a SM
      int ipt = TPdigi->id().iphi() + 2;
      if ( ipt > 72 ) ipt = ipt - 72;
      
      float xiet = (iet>0) ? iet-0.5 : iet+0.5 ;
      float xipt = ipt-0.5;

      if ( (TPdigi->ttFlag() & 0x3) == 0 ) {
        EBLowInterestTriggerTowerFlagMap_->Fill(xipt,xiet);
      } else {
	EBLowInterestTriggerTowerFlagMap_->Fill(-1,-18);
      }

      if ( (TPdigi->ttFlag() & 0x3) == 3 ) {
        EBHighInterestTriggerTowerFlagMap_->Fill(xipt,xiet);
      } else {
	EBHighInterestTriggerTowerFlagMap_->Fill(-1,-18);
      }

    }
  } else {
    LogWarning("EBSelectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }

  integral03 = h03->GetEntries();
  if( integral03 != 0 ) h03->Scale( 1.0/integral03 );
  integral04 = h04->GetEntries();
  if( integral04 != 0 ) h04->Scale( 1.0/integral04 );

  if (!ebSrFlags.isValid()) return;

  // Data Volume
  double aLowInterest=0;
  double aHighInterest=0;
  double aAnyInterest=0;

  Handle<EBDigiCollection> ebDigis;
  if ( e.getByLabel(EBDigiCollection_ , ebDigis) ) {

    anaDigiInit();

    for (unsigned int digis=0; digis<ebDigis->size(); ++digis){
      EBDataFrame ebdf = (*ebDigis)[digis];
      anaDigi(ebdf, *ebSrFlags);
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

  } else {
    LogWarning("EBSelectiveReadoutTask") << EBDigiCollection_ << " not available";
  }

}

void EBSelectiveReadoutTask::anaDigi(const EBDataFrame& frame, const EBSrFlagCollection& srFlagColl){

  EBDetId id = frame.id();
  EBSrFlagCollection::const_iterator srf = srFlagColl.find(readOutUnitOf(id));

  if(srf == srFlagColl.end()){
//    LogWarning("EBSelectiveReadoutTask") << "SR flag not found";
    return;
  }

  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
                       == EcalSrFlag::SRF_FULL);

  bool barrel = (id.subdetId()==EcalBarrel);

  if(barrel){
    ++nEb_;
    if(highInterest){
      ++nEbHI_;
    } else{//low interest
      ++nEbLI_;
    }
    int iEta0 = iEta2cIndex(id.ieta());
    int iPhi0 = iPhi2cIndex(id.iphi());
    if(!ebRuActive_[iEta0/ebTtEdge][iPhi0/ebTtEdge]){
      ++nRuPerDcc_[dccNum(id)-1];
      ebRuActive_[iEta0/ebTtEdge][iPhi0/ebTtEdge] = true;
    }
  }

  ++nPerDcc_[dccNum(id)-1];
}

void EBSelectiveReadoutTask::anaDigiInit(){
  nEb_ = 0;
  nEbLI_ = 0;
  nEbHI_ = 0;
  bzero(nPerDcc_, sizeof(nPerDcc_));
  bzero(nRuPerDcc_, sizeof(nRuPerDcc_));
  bzero(ebRuActive_, sizeof(ebRuActive_));
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

