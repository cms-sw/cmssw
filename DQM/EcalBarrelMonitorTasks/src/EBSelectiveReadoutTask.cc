#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <DQM/EcalCommon/interface/Numbers.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBSelectiveReadoutTask.h>

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;


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
  EcalFEDRawCollection_ = ps.getParameter<edm::InputTag>("EcalFEDRawCollection");

  // histograms...
  EcalDccEventSize_ = 0;
  EcalReadoutUnitForcedBitMap_ = 0;
  EcalFullReadoutSRFlagMap_ = 0;
  EcalHighInterestTriggerTowerFlagMap_ = 0;
  EcalLowInterestTriggerTowerFlagMap_ = 0;
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

  // endcap mapping
  edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
  c.get<IdealGeometryRecord>().get(hTriggerTowerMap);
  triggerTowerMap_ = hTriggerTowerMap.product();

}

void EBSelectiveReadoutTask::setup(void) {

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");

    sprintf(histo, "EBSRT Ecal DCC event size");
    EcalDccEventSize_ = dqmStore_->bookProfile(histo, histo, nECALDcc, 0, nECALDcc, 100, 0., 200., "s");
    EcalDccEventSize_->setAxisTitle("Event size (kB)",1);

    sprintf(histo, "EBSRT Ecal readout unit with SR forced");
    EcalReadoutUnitForcedBitMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EcalReadoutUnitForcedBitMap_->setAxisTitle("jphi", 1);
    EcalReadoutUnitForcedBitMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT Ecal full readout SR flags");
    EcalFullReadoutSRFlagMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EcalFullReadoutSRFlagMap_->setAxisTitle("jphi", 1);
    EcalFullReadoutSRFlagMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT Ecal high interest TT Flags");
    EcalHighInterestTriggerTowerFlagMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EcalHighInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EcalHighInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT Ecal low interest TT Flags");
    EcalLowInterestTriggerTowerFlagMap_ = dqmStore_->book2D(histo, histo, 72, 0, 72, 34, -17, 17);
    EcalLowInterestTriggerTowerFlagMap_->setAxisTitle("jphi", 1);
    EcalLowInterestTriggerTowerFlagMap_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBSRT EB event size");
    EBEventSize_ = dqmStore_->book1D(histo, histo, 100, 0, 200);
    EBEventSize_->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EBSRT EB high interest payload");
    EBHighInterestPayload_ =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EBHighInterestPayload_->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EBSRT EB low interest payload");
    EBLowInterestPayload_ =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EBLowInterestPayload_->setAxisTitle("event size (kB)",1);

  }

}

void EBSelectiveReadoutTask::cleanup(void){

  if ( ! init_ ) return;
  
  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBSelectiveReadoutTask");
    
    if ( EcalDccEventSize_ ) dqmStore_->removeElement( EcalDccEventSize_->getName() );
    EcalDccEventSize_ = 0;

    if ( EcalReadoutUnitForcedBitMap_ ) dqmStore_->removeElement( EcalReadoutUnitForcedBitMap_->getName() );
    EcalReadoutUnitForcedBitMap_ = 0;

    if ( EcalFullReadoutSRFlagMap_ ) dqmStore_->removeElement( EcalFullReadoutSRFlagMap_->getName() );
    EcalFullReadoutSRFlagMap_ = 0;

    if ( EcalHighInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EcalHighInterestTriggerTowerFlagMap_->getName() );
    EcalHighInterestTriggerTowerFlagMap_ = 0;

    if ( EcalLowInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EcalLowInterestTriggerTowerFlagMap_->getName() );
    EcalLowInterestTriggerTowerFlagMap_ = 0;

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
   
  if ( EcalDccEventSize_ ) EcalDccEventSize_->Reset();
  
  if ( EcalReadoutUnitForcedBitMap_ ) EcalReadoutUnitForcedBitMap_->Reset();

  if ( EcalFullReadoutSRFlagMap_ ) EcalFullReadoutSRFlagMap_->Reset();

  if ( EcalHighInterestTriggerTowerFlagMap_ ) EcalHighInterestTriggerTowerFlagMap_->Reset();

  if ( EcalLowInterestTriggerTowerFlagMap_ ) EcalLowInterestTriggerTowerFlagMap_->Reset();

  if ( EBEventSize_ ) EBEventSize_->Reset();

  if ( EBHighInterestPayload_ ) EBHighInterestPayload_->Reset();

  if ( EBLowInterestPayload_ ) EBLowInterestPayload_->Reset();
  
}
 
void EBSelectiveReadoutTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<FEDRawDataCollection> raw;
  if ( e.getByLabel(EcalFEDRawCollection_, raw) ) {

    for(int iDcc = 0; iDcc < nECALDcc; ++iDcc){

      EcalDccEventSize_->Fill(iDcc+1, ((double)raw->FEDData(601+iDcc).size())/kByte );
      
    }
  }
  else {
    LogWarning("EBSlectiveReadoutTask") << EcalFEDRawCollection_ << " not available";    
  }
  
  // Selective Readout Flags
  Handle<EBSrFlagCollection> ebSrFlags;
  if ( e.getByLabel(EBSRFlagCollection_,ebSrFlags) ) {
    
    for(EBSrFlagCollection::const_iterator it = ebSrFlags->begin();
	it != ebSrFlags->end(); ++it){
      const EBSrFlag& srf = *it;

      int iet = srf.id().ieta();
      int ipt = srf.id().iphi();

      float xiet = (iet>0) ? iet-0.5 : iet+0.5 ;
      float xipt = ipt-0.5;

      int flag = srf.value() & ~EcalSrFlag::SRF_FORCED_MASK;
      if(flag == EcalSrFlag::SRF_FULL){ 
	EcalFullReadoutSRFlagMap_->Fill(xipt,xiet);
      }
      if(srf.value() & EcalSrFlag::SRF_FORCED_MASK){
	EcalReadoutUnitForcedBitMap_->Fill(xipt,xiet);
      }
    }
  }
  else {
    LogWarning("EBSlectiveReadoutTask") << EBSRFlagCollection_ << " not available";
  }


  Handle<EcalTrigPrimDigiCollection> TPCollection;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, TPCollection) ) {
    
    // Trigger Primitives
    EcalTrigPrimDigiCollection::const_iterator TPdigi;
    for(TPdigi = TPCollection->begin(); TPdigi != TPCollection->end(); ++TPdigi) {

      EcalTriggerPrimitiveDigi data = (*TPdigi);
      EcalTrigTowerDetId idt = data.id();

      int iet = idt.ieta();
      int ipt = idt.iphi();

      float xiet = (iet>0) ? iet-0.5 : iet+0.5 ;
      float xipt = ipt-0.5;

      if ( (TPdigi->ttFlag() & 0x3) == 0 ) {
	EcalLowInterestTriggerTowerFlagMap_->Fill(xipt,xiet);
      }

      if ( (TPdigi->ttFlag() & 0x3) == 3 ) {
	EcalHighInterestTriggerTowerFlagMap_->Fill(xipt,xiet);
      }

    }
  }
  else {
    LogWarning("EBSlectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }


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
    
    //low interesest channels:
    aLowInterest = nEbLI_*bytesPerCrystal/kByte;
    EBLowInterestPayload_->Fill(aLowInterest);

    //low interesest channels:
    aHighInterest = nEbHI_*bytesPerCrystal/kByte;
    EBHighInterestPayload_->Fill(aHighInterest);

    //any-interest channels:
    aAnyInterest = getEbEventSize(nEb_)/kByte;
    EBEventSize_->Fill(aAnyInterest);

  }
  else {
    LogWarning("EBSlectiveReadoutTask") << EBDigiCollection_ << " not available";
  }


}


template<class T, class U>
void EBSelectiveReadoutTask::anaDigi(const T& frame,
				     const U& srFlagColl){
  const DetId& xtalId = frame.id();
  typename U::const_iterator srf = srFlagColl.find(readOutUnitOf(frame.id()));
  
  if(srf == srFlagColl.end()){
    throw cms::Exception("EBSelectiveReadoutTask")
      << __FILE__ << ":" << __LINE__ << ": SR flag not found";
  }
  
  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
		       == EcalSrFlag::SRF_FULL);
  
  bool barrel = (xtalId.subdetId()==EcalBarrel);

  if(barrel){
    ++nEb_;
    if(highInterest){
      ++nEbHI_;
    } else{//low interest
      ++nEbLI_;
    }
    int iEta0 = iEta2cIndex(static_cast<const EBDetId&>(xtalId).ieta());
    int iPhi0 = iPhi2cIndex(static_cast<const EBDetId&>(xtalId).iphi());
    if(!ebRuActive_[iEta0/ebTtEdge][iPhi0/ebTtEdge]){
      ++nRuPerDcc_[dccNum(xtalId)-1];
      ebRuActive_[iEta0/ebTtEdge][iPhi0/ebTtEdge] = true;
    }
  }

  ++nPerDcc_[dccNum(xtalId)-1];
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
  return triggerTowerMap_->towerOf(xtalId);
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
    throw cms::Exception("EBSelectiveReadoutTask")
      <<"Not recognized subdetector. Probably a bug.";
  }
  int iDcc0 = dccIndex(j,k);
  assert(iDcc0>=0 && iDcc0<nECALDcc);
  return iDcc0+1;
}

double EBSelectiveReadoutTask::getEbEventSize(double nReadXtals) const{
  double ruHeaderPayload = 0.;
  const int nEEDcc = 18;
  const int firstEbDcc0 = nEEDcc/2;
  for(int iDcc0 = firstEbDcc0; iDcc0 < firstEbDcc0 + nEBDcc; ++iDcc0){
    ruHeaderPayload += nRuPerDcc_[iDcc0]*8.;
  }
  
  return getDccOverhead(EB)*nEBDcc + nReadXtals*bytesPerCrystal
    + ruHeaderPayload;
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

