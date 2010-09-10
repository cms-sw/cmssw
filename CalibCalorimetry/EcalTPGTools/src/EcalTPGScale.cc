#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"


EcalTPGScale::EcalTPGScale()
{ }

EcalTPGScale::~EcalTPGScale()
{ }

void EcalTPGScale::setEventSetup(const edm::EventSetup & evtSetup)
{
  setup_ = &evtSetup ;
}

double EcalTPGScale::getTPGInGeV(const EcalTriggerPrimitiveDigi & tpDigi)
{ 
   const EcalTrigTowerDetId & towerId = tpDigi.id() ;
   int ADC = tpDigi.compressedEt() ;
   return getTPGInGeV(ADC, towerId) ;
}

double EcalTPGScale::getTPGInGeV(unsigned int ADC, const EcalTrigTowerDetId & towerId)
{ 
  // 1. get lsb
  edm::ESHandle<EcalTPGPhysicsConst> physHandle;
  setup_->get<EcalTPGPhysicsConstRcd>().get( physHandle );
  const EcalTPGPhysicsConstMap & physMap = physHandle.product()->getMap() ;

  uint32_t eb = DetId(DetId::Ecal,EcalBarrel).rawId() ;
  uint32_t ee = DetId(DetId::Ecal,EcalEndcap).rawId() ;
  EcalTPGPhysicsConstMapIterator it = physMap.end() ;
  if (towerId.subDet() == EcalBarrel) it = physMap.find(eb) ;
  else if (towerId.subDet() == EcalEndcap) it = physMap.find(ee) ;
  double lsb10bits = 0. ;
  if (it != physMap.end()) {
    EcalTPGPhysicsConst::Item item = it->second ;
    lsb10bits = item.EtSat/1024. ;
  }

  // 2. linearized TPG
  return lsb10bits * getLinearizedTPG(ADC, towerId) ;

}

unsigned int EcalTPGScale::getLinearizedTPG(unsigned int ADC, const EcalTrigTowerDetId & towerId)
{
  int tpg10bits = 0 ;
 
  // Get compressed look-up table
  edm::ESHandle<EcalTPGLutGroup> lutGrpHandle;
  setup_->get<EcalTPGLutGroupRcd>().get( lutGrpHandle );
  const EcalTPGGroups::EcalTPGGroupsMap & lutGrpMap = lutGrpHandle.product()->getMap() ;  
  EcalTPGGroups::EcalTPGGroupsMapItr itgrp = lutGrpMap.find(towerId.rawId()) ;
  uint32_t lutGrp = 999 ;
  if (itgrp != lutGrpMap.end()) lutGrp = itgrp->second ;

  edm::ESHandle<EcalTPGLutIdMap> lutMapHandle;
  setup_->get<EcalTPGLutIdMapRcd>().get( lutMapHandle );
  const EcalTPGLutIdMap::EcalTPGLutMap & lutMap = lutMapHandle.product()->getMap() ;  
  EcalTPGLutIdMap::EcalTPGLutMapItr itLut = lutMap.find(lutGrp) ;
  if (itLut != lutMap.end()) {
    const unsigned int * lut = (itLut->second).getLut() ;
    for (unsigned int i=0 ; i<1024 ; i++)
      if (ADC == (0xff & lut[i])) {
	tpg10bits = i ;
	break ;
      }
  }

  return tpg10bits ;
}

unsigned int EcalTPGScale::getTPGInADC(double energy, const EcalTrigTowerDetId & towerId)
{
  unsigned int tpgADC = 0 ;

  // 1. get lsb
  edm::ESHandle<EcalTPGPhysicsConst> physHandle;
  setup_->get<EcalTPGPhysicsConstRcd>().get( physHandle );
  const EcalTPGPhysicsConstMap & physMap = physHandle.product()->getMap() ;

  uint32_t eb = DetId(DetId::Ecal,EcalBarrel).rawId() ;
  uint32_t ee = DetId(DetId::Ecal,EcalEndcap).rawId() ;
  EcalTPGPhysicsConstMapIterator it = physMap.end() ;
  if (towerId.subDet() == EcalBarrel) it = physMap.find(eb) ;
  else if (towerId.subDet() == EcalEndcap) it = physMap.find(ee) ;
  double lsb10bits = 0. ;
  if (it != physMap.end()) {
    EcalTPGPhysicsConst::Item item = it->second ;
    lsb10bits = item.EtSat/1024. ;
  }

  // 2. get compressed look-up table
  edm::ESHandle<EcalTPGLutGroup> lutGrpHandle;
  setup_->get<EcalTPGLutGroupRcd>().get( lutGrpHandle );
  const EcalTPGGroups::EcalTPGGroupsMap & lutGrpMap = lutGrpHandle.product()->getMap() ;  
  EcalTPGGroups::EcalTPGGroupsMapItr itgrp = lutGrpMap.find(towerId) ;
  uint32_t lutGrp = 0 ;
  if (itgrp != lutGrpMap.end()) lutGrp = itgrp->second ;
  
  edm::ESHandle<EcalTPGLutIdMap> lutMapHandle;
  setup_->get<EcalTPGLutIdMapRcd>().get( lutMapHandle );
  const EcalTPGLutIdMap::EcalTPGLutMap & lutMap = lutMapHandle.product()->getMap() ;  
  EcalTPGLutIdMap::EcalTPGLutMapItr itLut = lutMap.find(lutGrp) ;
  if (itLut != lutMap.end()) {
    const unsigned int * lut = (itLut->second).getLut() ;
    if (lsb10bits>0) {
      int tpgADC10b = int(energy/lsb10bits+0.5) ;
      if (tpgADC10b>=0 && tpgADC10b<1024) tpgADC = (0xff & lut[tpgADC10b]) ;
      if (tpgADC10b>=1024) tpgADC = 0xff ;
    }
  }

  return tpgADC ;
}
