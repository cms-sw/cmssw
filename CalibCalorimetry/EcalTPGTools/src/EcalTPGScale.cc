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

double EcalTPGScale::getTPGInGeV(const edm::EventSetup & evtSetup, const EcalTriggerPrimitiveDigi & tpDigi)
{ 
   const EcalTrigTowerDetId & towerId = tpDigi.id() ;
   int ADC = tpDigi.compressedEt() ;
   return getTPGInGeV(evtSetup, ADC, towerId) ;
}

double EcalTPGScale::getTPGInGeV(const edm::EventSetup & evtSetup, int ADC, const EcalTrigTowerDetId & towerId)
{ 
  double tpgInGev = 0. ;

  // 1. get lsb
  edm::ESHandle<EcalTPGPhysicsConst> physHandle;
  evtSetup.get<EcalTPGPhysicsConstRcd>().get( physHandle );
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
  evtSetup.get<EcalTPGLutGroupRcd>().get( lutGrpHandle );
  const EcalTPGGroups::EcalTPGGroupsMap & lutGrpMap = lutGrpHandle.product()->getMap() ;  
  EcalTPGGroups::EcalTPGGroupsMapItr itgrp = lutGrpMap.find(towerId) ;
  uint32_t lutGrp = 0 ;
  if (itgrp != lutGrpMap.end()) lutGrp = itgrp->second ;
  
  edm::ESHandle<EcalTPGLutIdMap> lutMapHandle;
  evtSetup.get<EcalTPGLutIdMapRcd>().get( lutMapHandle );
  const EcalTPGLutIdMap::EcalTPGLutMap & lutMap = lutMapHandle.product()->getMap() ;  
  EcalTPGLutIdMap::EcalTPGLutMapItr itLut = lutMap.find(lutGrp) ;
  if (itLut != lutMap.end()) {
    const unsigned int * lut = (itLut->second).getLut() ;
    for (uint i=0 ; i<1024 ; i++)
      if (ADC == lut[i]) {
	tpgInGev = i*lsb10bits ;
	break ;
      }
  }

  return tpgInGev ;
}

int EcalTPGScale::getTPGInADC(const edm::EventSetup & evtSetup, double energy, const EcalTrigTowerDetId & towerId)
{
  return 0 ;
}
