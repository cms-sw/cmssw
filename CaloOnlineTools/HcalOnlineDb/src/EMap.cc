// -*- C++ -*-
//
// Package: CaloOnlineTools/HcalOnlineDb
// Class:   EMap
//
/**\class WriteL1TriggerObjectsTxt WriteL1TriggerObjectsTxt.cc CaloOnlineTools/HcalOnlineDb/WriteL1TriggerObjectsTxt.cc

 Description: Holds a custom struct of precision and trigger channel electronics map info

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 23 14:30:20 CDT 2007
//

#include "CaloOnlineTools/HcalOnlineDb/interface/EMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

EMap::EMap(const HcalElectronicsMap* emap) {
  if (emap) {
    std::vector<HcalElectronicsId> v_eId = emap->allElectronicsIdPrecision();
    for (std::vector<HcalElectronicsId>::const_iterator eId = v_eId.begin(); eId != v_eId.end(); eId++) {
      EMapRow row;
      row.crate = eId->readoutVMECrateId();
      row.slot = eId->htrSlot();
      row.dcc = eId->dccid();
      row.spigot = eId->spigot();
      row.fiber = eId->fiberIndex();
      row.fiberchan = eId->fiberChanId();
      if (eId->htrTopBottom() == 1)
        row.topbottom = "t";
      else if (eId->htrTopBottom() == 0)
        row.topbottom = "b";
      else
        row.topbottom = "u";
      HcalGenericDetId _gid(emap->lookup(*eId));
      if (!(_gid.null()) && (_gid.genericSubdet() == HcalGenericDetId::HcalGenBarrel ||
                             _gid.genericSubdet() == HcalGenericDetId::HcalGenEndcap ||
                             _gid.genericSubdet() == HcalGenericDetId::HcalGenForward ||
                             _gid.genericSubdet() == HcalGenericDetId::HcalGenOuter)) {
        HcalDetId _id(emap->lookup(*eId));
        row.rawId = _id.rawId();
        row.ieta = _id.ieta();
        row.iphi = _id.iphi();
        row.idepth = _id.depth();
        row.subdet = getSubdetectorString(_id.subdet());
        map.push_back(row);
      } else if (!(_gid.null()) && _gid.genericSubdet() == HcalGenericDetId::HcalGenZDC) {
        HcalZDCDetId _id(emap->lookup(*eId));
        row.zdc_channel = _id.channel();
        row.zdc_section = getZDCSectionString(_id.section());
        row.idepth = _id.depth();
        row.zdc_zside = _id.zside();
        map.push_back(row);
      }
    }

    v_eId = emap->allElectronicsIdTrigger();
    for (std::vector<HcalElectronicsId>::const_iterator eId = v_eId.begin(); eId != v_eId.end(); eId++) {
      EMapRow row;
      row.crate = eId->readoutVMECrateId();
      row.slot = eId->htrSlot();
      row.dcc = eId->dccid();
      row.spigot = eId->spigot();
      row.fiber = eId->isVMEid() ? eId->slbSiteNumber() : eId->fiberIndex();
      row.fiberchan = eId->isVMEid() ? eId->slbChannelIndex() : eId->fiberChanId();
      if (eId->htrTopBottom() == 1)
        row.topbottom = "t";
      else if (eId->htrTopBottom() == 0)
        row.topbottom = "b";
      else
        row.topbottom = "u";
      HcalTrigTowerDetId _id(emap->lookupTrigger(*eId));
      if (!(_id.null())) {
        row.rawId = _id.rawId();
        row.ieta = _id.ieta();
        row.iphi = _id.iphi();
        row.idepth = _id.depth();
        row.subdet = getSubdetectorString(_id.subdet());
        map.push_back(row);
      }
    }
  } else {
    edm::LogError("EMap") << "Pointer to HcalElectronicsMap is 0!!!";
  }
}

std::string EMap::getSubdetectorString(const HcalSubdetector& _det) {
  std::string sDet;
  if (_det == HcalBarrel)
    sDet = "HB";
  else if (_det == HcalEndcap)
    sDet = "HE";
  else if (_det == HcalForward)
    sDet = "HF";
  else if (_det == HcalOuter)
    sDet = "HO";
  else if (_det == HcalTriggerTower)
    sDet = "HT";
  else
    sDet = "other";
  return sDet;
}

std::string EMap::getZDCSectionString(const HcalZDCDetId::Section& _section) {
  std::string zdcSection;
  if (_section == HcalZDCDetId::EM)
    zdcSection = "ZDC EM";
  else if (_section == HcalZDCDetId::HAD)
    zdcSection = "ZDC HAD";
  else if (_section == HcalZDCDetId::LUM)
    zdcSection = "ZDC LUM";
  else if (_section == HcalZDCDetId::RPD)
    zdcSection = "ZDC RPD";
  else
    zdcSection = "UNKNOWN";
  return zdcSection;
}
