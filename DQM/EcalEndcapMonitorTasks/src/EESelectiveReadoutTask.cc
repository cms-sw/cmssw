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
//#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <DQM/EcalCommon/interface/Numbers.h>
#include <DQM/EcalEndcapMonitorTasks/interface/EESelectiveReadoutTask.h>

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;


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
  EcalFEDRawCollection_ = ps.getParameter<edm::InputTag>("EcalFEDRawCollection");

  // histograms...
  EEReadoutUnitForcedBitMap_ = 0;
  EEFullReadoutSRFlagMap_ = 0;
  EEHighInterestTriggerTowerFlagMap_ = 0;
  EELowInterestTriggerTowerFlagMap_ = 0;
  EEEventSize_ = 0;
  EEHighInterestPayload_ = 0;
  EELowInterestPayload_ = 0;

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

  // endcap mapping
//   edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
//   c.get<IdealGeometryRecord>().get(hTriggerTowerMap);
//   triggerTowerMap_ = hTriggerTowerMap.product();

}

void EESelectiveReadoutTask::setup(void) {

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");

    sprintf(histo, "EESRT EE readout unit with SR forced");
    EEReadoutUnitForcedBitMap_ = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEReadoutUnitForcedBitMap_->setAxisTitle("jx", 1);
    EEReadoutUnitForcedBitMap_->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT EE full readout SR flags");
    EEFullReadoutSRFlagMap_ = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEFullReadoutSRFlagMap_->setAxisTitle("jx", 1);
    EEFullReadoutSRFlagMap_->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT EE high interest TT Flags");
    EEHighInterestTriggerTowerFlagMap_ = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EEHighInterestTriggerTowerFlagMap_->setAxisTitle("jx", 1);
    EEHighInterestTriggerTowerFlagMap_->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT EE low interest TT Flags");
    EELowInterestTriggerTowerFlagMap_ = dqmStore_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    EELowInterestTriggerTowerFlagMap_->setAxisTitle("jx", 1);
    EELowInterestTriggerTowerFlagMap_->setAxisTitle("jy", 2);

    sprintf(histo, "EESRT EE event size");
    EEEventSize_ = dqmStore_->book1D(histo, histo, 100, 0, 200);
    EEEventSize_->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT EE high interest payload");
    EEHighInterestPayload_ =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EEHighInterestPayload_->setAxisTitle("event size (kB)",1);

    sprintf(histo, "EESRT EE low interest payload");
    EELowInterestPayload_ =  dqmStore_->book1D(histo, histo, 100, 0, 200);
    EELowInterestPayload_->setAxisTitle("event size (kB)",1);

  }

}

void EESelectiveReadoutTask::cleanup(void){

  if ( ! init_ ) return;
  
  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EESelectiveReadoutTask");
    
    if ( EEReadoutUnitForcedBitMap_ ) dqmStore_->removeElement( EEReadoutUnitForcedBitMap_->getName() );
    EEReadoutUnitForcedBitMap_ = 0;

    if ( EEFullReadoutSRFlagMap_ ) dqmStore_->removeElement( EEFullReadoutSRFlagMap_->getName() );
    EEFullReadoutSRFlagMap_ = 0;

    if ( EEHighInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EEHighInterestTriggerTowerFlagMap_->getName() );
    EEHighInterestTriggerTowerFlagMap_ = 0;

    if ( EELowInterestTriggerTowerFlagMap_ ) dqmStore_->removeElement( EELowInterestTriggerTowerFlagMap_->getName() );
    EELowInterestTriggerTowerFlagMap_ = 0;

    if ( EEEventSize_ ) dqmStore_->removeElement( EEEventSize_->getName() );
    EEEventSize_ = 0;

    if ( EEHighInterestPayload_ ) dqmStore_->removeElement( EEHighInterestPayload_->getName() );
    EEHighInterestPayload_ = 0;

    if ( EELowInterestPayload_ ) dqmStore_->removeElement( EELowInterestPayload_->getName() );
    EELowInterestPayload_ = 0;

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
   
  if ( EEReadoutUnitForcedBitMap_ ) EEReadoutUnitForcedBitMap_->Reset();

  if ( EEFullReadoutSRFlagMap_ ) EEFullReadoutSRFlagMap_->Reset();

  if ( EEHighInterestTriggerTowerFlagMap_ ) EEHighInterestTriggerTowerFlagMap_->Reset();

  if ( EELowInterestTriggerTowerFlagMap_ ) EELowInterestTriggerTowerFlagMap_->Reset();

  if ( EEEventSize_ ) EEEventSize_->Reset();

  if ( EEHighInterestPayload_ ) EEHighInterestPayload_->Reset();

  if ( EELowInterestPayload_ ) EELowInterestPayload_->Reset();
  
}
 
void EESelectiveReadoutTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // Selective Readout Flags
  Handle<EESrFlagCollection> eeSrFlags;
  if ( e.getByLabel(EESRFlagCollection_,eeSrFlags) ) {
    
    for(EESrFlagCollection::const_iterator it = eeSrFlags->begin();
	it != eeSrFlags->end(); ++it){
      const EESrFlag& srf = *it;

      int ix = srf.id().ix();
      int iy = srf.id().iy();

      int zside = srf.id().zside();

      if ( zside < 0 ) ix = 101 - ix;

      float xix = ix-0.5;
      float xiy = iy-0.5;

      int flag = srf.value() & ~EcalSrFlag::SRF_FORCED_MASK;
      if(flag == EcalSrFlag::SRF_FULL){ 
	EEFullReadoutSRFlagMap_->Fill(xix,xiy);
      }
      if(srf.value() & EcalSrFlag::SRF_FORCED_MASK){
	EEReadoutUnitForcedBitMap_->Fill(xix,xiy);
      }
    }
  }
  else {
    LogWarning("EESlectiveReadoutTask") << EESRFlagCollection_ << " not available";
  }


  Handle<EcalTrigPrimDigiCollection> TPCollection;
  if ( e.getByLabel(EcalTrigPrimDigiCollection_, TPCollection) ) {
    
    // Trigger Primitives
    EcalTrigPrimDigiCollection::const_iterator TPdigi;
    for(TPdigi = TPCollection->begin(); TPdigi != TPCollection->end(); ++TPdigi) {

      EcalTriggerPrimitiveDigi data = (*TPdigi);
      EcalTrigTowerDetId idt = data.id();

      int ismt = Numbers::iSM( idt );

      vector<DetId> crystals = Numbers::crystals( idt );
      
      for ( unsigned int i=0; i<crystals.size(); i++ ) {
	
	EEDetId id = crystals[i];
	
	int ix = id.ix();
	int iy = id.iy();
	
	if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

	float xix = ix-0.5;
	float xiy = iy-0.5;
	
	if ( (TPdigi->ttFlag() & 0x3) == 0 ) {
	  EELowInterestTriggerTowerFlagMap_->Fill(xix,xiy);
	}
	
	if ( (TPdigi->ttFlag() & 0x3) == 3 ) {
	  EEHighInterestTriggerTowerFlagMap_->Fill(xix,xiy);
	}
      }

    }
  }
  else {
    LogWarning("EESlectiveReadoutTask") << EcalTrigPrimDigiCollection_ << " not available";
  }


  // Data Volume
  double aLowInterest=0;
  double aHighInterest=0;
  double aAnyInterest=0;

  Handle<EEDigiCollection> eeDigis;
  if ( e.getByLabel(EEDigiCollection_ , eeDigis) ) {
    
    anaDigiInit();
    
    for (unsigned int digis=0; digis<eeDigis->size(); ++digis){
      EEDataFrame eedf = (*eeDigis)[digis];
      anaDigi(eedf, *eeSrFlags);
    }
    
    //low interesest channels:
    aLowInterest = nEeLI_*bytesPerCrystal/kByte;
    EELowInterestPayload_->Fill(aLowInterest);

    //low interesest channels:
    aHighInterest = nEeHI_*bytesPerCrystal/kByte;
    EEHighInterestPayload_->Fill(aHighInterest);

    //any-interest channels:
    aAnyInterest = getEeEventSize(nEe_)/kByte;
    EEEventSize_->Fill(aAnyInterest);

  }
  else {
    LogWarning("EESlectiveReadoutTask") << EEDigiCollection_ << " not available";
  }


}


template<class T, class U>
void EESelectiveReadoutTask::anaDigi(const T& frame,
				     const U& srFlagColl){
  const DetId& xtalId = frame.id();
  typename U::const_iterator srf = srFlagColl.find(readOutUnitOf(frame.id()));
  
  if(srf == srFlagColl.end()){
    throw cms::Exception("EESelectiveReadoutTask")
      << __FILE__ << ":" << __LINE__ << ": SR flag not found";
  }
  
  bool highInterest = ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK)
		       == EcalSrFlag::SRF_FULL);
  
  bool endcap = (xtalId.subdetId()==EcalEndcap);

  if(endcap){
    ++nEe_;
    if(highInterest){
      ++nEeHI_;
    } else{//low interest
      ++nEeLI_;
    }
    int iX0 = iXY2cIndex(static_cast<const EEDetId&>(frame.id()).ix());
    int iY0 = iXY2cIndex(static_cast<const EEDetId&>(frame.id()).iy());
    int iZ0 = static_cast<const EEDetId&>(frame.id()).zside()>0?1:0;
    
    if(!eeRuActive_[iZ0][iX0/scEdge][iY0/scEdge]){
      ++nRuPerDcc_[dccNum(xtalId)];
      eeRuActive_[iZ0][iX0/scEdge][iY0/scEdge] = true;
    }
  }

  ++nPerDcc_[dccNum(xtalId)-1];
}

void EESelectiveReadoutTask::anaDigiInit(){
  nEe_ = 0;
  nEeLI_ = 0;
  nEeHI_ = 0;
  bzero(nPerDcc_, sizeof(nPerDcc_));
  bzero(nRuPerDcc_, sizeof(nRuPerDcc_));
  bzero(eeRuActive_, sizeof(eeRuActive_));
}

EcalScDetId
EESelectiveReadoutTask::superCrystalOf(const EEDetId& xtalId) const
{
  const int scEdge = 5;
  return EcalScDetId((xtalId.ix()-1)/scEdge+1,
		     (xtalId.iy()-1)/scEdge+1,
		     xtalId.zside());
}

EcalScDetId
EESelectiveReadoutTask::readOutUnitOf(const EEDetId& xtalId) const{
  return superCrystalOf(xtalId);
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
  for(int iDcc0 = 0; iDcc0 < nECALDcc; ++iDcc0){
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

