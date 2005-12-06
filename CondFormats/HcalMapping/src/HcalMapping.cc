#include "CondFormats/HcalMapping/interface/HcalMapping.h"
#include <iostream>

HcalMapping::HcalMapping(bool maintainL2E) : 
  maintainL2E_(maintainL2E), 
  elecToLogical_(HcalElectronicsId::maxLinearIndex+1,0),
  elecToTrigTower_(HcalElectronicsId::maxLinearIndex+1,0) {
  clear();
}

void HcalMapping::clear() {
  logicalToElec_.clear();
  trigTowerToElec_.clear();
  elecToLogical_=std::vector<HcalDetId>(HcalElectronicsId::maxLinearIndex+1,0);
  elecToTrigTower_=std::vector<HcalTrigTowerDetId>(HcalElectronicsId::maxLinearIndex+1,0);
  for (int i=0; i<=HcalElectronicsId::maxDCCId; i++) {
    dccIds_[i].hbheEntries=0;
    dccIds_[i].hfEntries=0;
    dccIds_[i].hoEntries=0;
  }
}

std::vector<HcalDetId>::const_iterator HcalMapping::detid_begin() const {
  return elecToLogical_.begin();
}

std::vector<HcalDetId>::const_iterator HcalMapping::detid_end() const {
  return elecToLogical_.end();
}

std::vector<HcalTrigTowerDetId>::const_iterator HcalMapping::trigger_detid_begin() const {
  return elecToTrigTower_.begin();
}

std::vector<HcalTrigTowerDetId>::const_iterator HcalMapping::trigger_detid_end() const {
  return elecToTrigTower_.end();
}

const HcalDetId& HcalMapping::lookup(const HcalElectronicsId& el) const {
  const HcalDetId& glog=elecToLogical_[el.linearIndex()];
  //  if (glog.rawId()==0) throw new std::exception(); // no such item
  return glog;
}

static const HcalElectronicsId nullEID;

const HcalElectronicsId& HcalMapping::lookup(const HcalDetId& id) const {
  std::map<HcalDetId,HcalElectronicsId>::const_iterator i=logicalToElec_.find(id);
  if (i==logicalToElec_.end()) return nullEID;
  return i->second;
}

const HcalTrigTowerDetId& HcalMapping::lookupTrigger(const HcalElectronicsId& el) const {
  const HcalTrigTowerDetId& glog=elecToTrigTower_[el.linearIndex()];
  //  if (glog.rawId()==0) throw new std::exception(); // no such item
  return glog;
}

const HcalElectronicsId& HcalMapping::lookupTrigger(const HcalTrigTowerDetId& id) const {
  std::map<HcalTrigTowerDetId,HcalElectronicsId>::const_iterator i=trigTowerToElec_.find(id);
  if (i==trigTowerToElec_.end()) return nullEID;
  return i->second;
}


void HcalMapping::setTriggerMap(const HcalElectronicsId& eid, const HcalTrigTowerDetId& lid) {
  elecToTrigTower_[eid.linearIndex()]=lid;
  if (maintainL2E_) trigTowerToElec_.insert(std::pair<HcalTrigTowerDetId,HcalElectronicsId>(lid,eid));
}

void HcalMapping::setMap(const HcalElectronicsId& eid, const HcalDetId& lid) {
  elecToLogical_[eid.linearIndex()]=lid;
  if (maintainL2E_) logicalToElec_.insert(std::pair<HcalDetId,HcalElectronicsId>(lid,eid));

  // check the effect on the DCC identification
  if (eid.dccid()>HcalElectronicsId::maxDCCId) return;
  int dcc=eid.dccid();

  if (lid.subdet()==HcalBarrel || lid.subdet()==HcalEndcap) dccIds_[dcc].hbheEntries++;
  if (lid.subdet()==HcalForward) dccIds_[dcc].hfEntries++;
  if (lid.subdet()==HcalOuter) dccIds_[dcc].hoEntries++;

}

bool HcalMapping::subdetectorPresent(HcalSubdetector det, int dccid) const {
  
  if (dccid<0 || dccid>HcalElectronicsId::maxDCCId) return false;
  switch (det) {
  case (HcalBarrel) :
  case (HcalEndcap) : return dccIds_[dccid].hbheEntries!=0;
  case (HcalOuter) : return dccIds_[dccid].hoEntries!=0;
  case (HcalForward) : return dccIds_[dccid].hfEntries!=0;
  default: return false;
  }
}

