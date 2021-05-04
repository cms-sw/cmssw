#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

#include "FWCore/Framework/interface/ESHandle.h"

EcalTPGScale::Tokens::Tokens(edm::ConsumesCollector c)
    : physConstToken_(c.esConsumes<EcalTPGPhysicsConst, EcalTPGPhysicsConstRcd>()),
      lutGrpToken_(c.esConsumes<EcalTPGLutGroup, EcalTPGLutGroupRcd>()),
      lutMapToken_(c.esConsumes<EcalTPGLutIdMap, EcalTPGLutIdMapRcd>()) {}

EcalTPGScale::EcalTPGScale(Tokens const& tokens, const edm::EventSetup& evtSetup)
    : phys_(evtSetup.getData(tokens.physConstToken_)),
      lutGrp_(evtSetup.getData(tokens.lutGrpToken_)),
      lut_(evtSetup.getData(tokens.lutMapToken_))

{}

double EcalTPGScale::getTPGInGeV(const EcalTriggerPrimitiveDigi& tpDigi) const {
  const EcalTrigTowerDetId& towerId = tpDigi.id();
  int ADC = tpDigi.compressedEt();
  return getTPGInGeV(ADC, towerId);
}

double EcalTPGScale::getTPGInGeV(unsigned int ADC, const EcalTrigTowerDetId& towerId) const {
  // 1. get lsb

  const EcalTPGPhysicsConstMap& physMap = phys_.getMap();
  uint32_t eb = DetId(DetId::Ecal, EcalBarrel).rawId();
  uint32_t ee = DetId(DetId::Ecal, EcalEndcap).rawId();
  EcalTPGPhysicsConstMapIterator it = physMap.end();
  if (towerId.subDet() == EcalBarrel)
    it = physMap.find(eb);
  else if (towerId.subDet() == EcalEndcap)
    it = physMap.find(ee);
  double lsb10bits = 0.;
  if (it != physMap.end()) {
    EcalTPGPhysicsConst::Item item = it->second;
    lsb10bits = item.EtSat / 1024.;
  }

  // 2. linearized TPG
  return lsb10bits * getLinearizedTPG(ADC, towerId);
}

unsigned int EcalTPGScale::getLinearizedTPG(unsigned int ADC, const EcalTrigTowerDetId& towerId) const {
  int tpg10bits = 0;

  const EcalTPGGroups::EcalTPGGroupsMap& lutGrpMap = lutGrp_.getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr itgrp = lutGrpMap.find(towerId.rawId());
  uint32_t lutGrp = 999;
  if (itgrp != lutGrpMap.end())
    lutGrp = itgrp->second;

  const EcalTPGLutIdMap::EcalTPGLutMap& lutMap = lut_.getMap();
  EcalTPGLutIdMap::EcalTPGLutMapItr itLut = lutMap.find(lutGrp);
  if (itLut != lutMap.end()) {
    const unsigned int* lut = (itLut->second).getLut();
    for (unsigned int i = 0; i < 1024; i++)
      if (ADC == (0xff & lut[i])) {
        tpg10bits = i;
        break;
      }
  }

  return tpg10bits;
}

unsigned int EcalTPGScale::getTPGInADC(double energy, const EcalTrigTowerDetId& towerId) const {
  unsigned int tpgADC = 0;

  // 1. get lsb

  const EcalTPGPhysicsConstMap& physMap = phys_.getMap();

  uint32_t eb = DetId(DetId::Ecal, EcalBarrel).rawId();
  uint32_t ee = DetId(DetId::Ecal, EcalEndcap).rawId();
  EcalTPGPhysicsConstMapIterator it = physMap.end();
  if (towerId.subDet() == EcalBarrel)
    it = physMap.find(eb);
  else if (towerId.subDet() == EcalEndcap)
    it = physMap.find(ee);
  double lsb10bits = 0.;
  if (it != physMap.end()) {
    EcalTPGPhysicsConst::Item item = it->second;
    lsb10bits = item.EtSat / 1024.;
  }

  // 2. get compressed look-up table

  const EcalTPGGroups::EcalTPGGroupsMap& lutGrpMap = lutGrp_.getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr itgrp = lutGrpMap.find(towerId);
  uint32_t lutGrp = 0;
  if (itgrp != lutGrpMap.end())
    lutGrp = itgrp->second;

  const EcalTPGLutIdMap::EcalTPGLutMap& lutMap = lut_.getMap();
  EcalTPGLutIdMap::EcalTPGLutMapItr itLut = lutMap.find(lutGrp);
  if (itLut != lutMap.end()) {
    const unsigned int* lut = (itLut->second).getLut();
    if (lsb10bits > 0) {
      int tpgADC10b = int(energy / lsb10bits + 0.5);
      if (tpgADC10b >= 0 && tpgADC10b < 1024)
        tpgADC = (0xff & lut[tpgADC10b]);
      if (tpgADC10b >= 1024)
        tpgADC = 0xff;
    }
  }

  return tpgADC;
}
