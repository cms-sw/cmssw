/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Hcal channels
$Author: ratnikov
$Date: 2007/09/18 06:48:38 $
$Revision: 1.22 $
*/

#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalObjectAddons.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalElectronicsMap::HcalElectronicsMap(const HcalElectronicsMapAddons::Helper& helper)
    : mPItems(helper.mPItems), mTItems(helper.mTItems) {
  initialize();
}

HcalElectronicsMap::~HcalElectronicsMap() {}
// copy-ctor
HcalElectronicsMap::HcalElectronicsMap(const HcalElectronicsMap& src)
    : mPItems(src.mPItems), mTItems(src.mTItems), mPItemsById(src.mPItemsById), mTItemsByTrigId(src.mTItemsByTrigId) {}
// copy assignment operator
HcalElectronicsMap& HcalElectronicsMap::operator=(const HcalElectronicsMap& rhs) {
  HcalElectronicsMap temp(rhs);
  temp.swap(*this);
  return *this;
}
// public swap function
void HcalElectronicsMap::swap(HcalElectronicsMap& other) {
  std::swap(mPItems, other.mPItems);
  std::swap(mTItems, other.mTItems);
  std::swap(mPItemsById, other.mPItemsById);
  std::swap(mTItemsByTrigId, other.mTItemsByTrigId);
}
// move constructor
HcalElectronicsMap::HcalElectronicsMap(HcalElectronicsMap&& other) { other.swap(*this); }

const HcalElectronicsMap::PrecisionItem* HcalElectronicsMap::findById(unsigned long fId) const {
  PrecisionItem target(fId, 0);
  return HcalObjectAddons::findByT<PrecisionItem, HcalElectronicsMapAddons::LessById>(&target, mPItemsById);
}

const HcalElectronicsMap::PrecisionItem* HcalElectronicsMap::findPByElId(unsigned long fElId) const {
  HcalElectronicsId eid(fElId);
  const PrecisionItem* i = &(mPItems[eid.linearIndex()]);

  if (i != nullptr && i->mElId != fElId)
    i = nullptr;
  return i;
}

const HcalElectronicsMap::TriggerItem* HcalElectronicsMap::findTByElId(unsigned long fElId) const {
  HcalElectronicsId eid(fElId);
  const TriggerItem* i = &(mTItems[eid.linearIndex()]);

  if (i != nullptr && i->mElId != fElId)
    i = nullptr;
  return i;
}

const HcalElectronicsMap::TriggerItem* HcalElectronicsMap::findByTrigId(unsigned long fTrigId) const {
  TriggerItem target(fTrigId, 0);
  return HcalObjectAddons::findByT<TriggerItem, HcalElectronicsMapAddons::LessByTrigId>(&target, mTItemsByTrigId);
}

const DetId HcalElectronicsMap::lookup(HcalElectronicsId fId) const {
  const PrecisionItem* item = findPByElId(fId.rawId());
  return DetId(item ? item->mId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookup(DetId fId) const {
  const PrecisionItem* item = findById(fId.rawId());
  return HcalElectronicsId(item ? item->mElId : 0);
}

const DetId HcalElectronicsMap::lookupTrigger(HcalElectronicsId fId) const {
  const TriggerItem* item = findTByElId(fId.rawId());
  return DetId(item ? item->mTrigId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookupTrigger(DetId fId) const {
  const TriggerItem* item = findByTrigId(fId.rawId());
  return HcalElectronicsId(item ? item->mElId : 0);
}

bool HcalElectronicsMap::lookup(const HcalElectronicsId pid, HcalElectronicsId& eid, HcalGenericDetId& did) const {
  const PrecisionItem* i = &(mPItems[pid.linearIndex()]);
  if (i != nullptr && i->mId != 0) {
    eid = HcalElectronicsId(i->mElId);
    did = HcalGenericDetId(i->mId);
    return true;
  } else
    return false;
}

bool HcalElectronicsMap::lookup(const HcalElectronicsId pid, HcalElectronicsId& eid, HcalTrigTowerDetId& did) const {
  const TriggerItem* i = &(mTItems[pid.linearIndex()]);
  if (i != nullptr && i->mTrigId != 0) {
    eid = HcalElectronicsId(i->mElId);
    did = HcalGenericDetId(i->mTrigId);
    return true;
  } else
    return false;
}

std::vector<HcalElectronicsId> HcalElectronicsMap::allElectronicsId() const {
  std::vector<HcalElectronicsId> result;
  for (auto mPItem : mPItems)
    if (mPItem.mElId)
      result.push_back(HcalElectronicsId(mPItem.mElId));
  for (auto mTItem : mTItems)
    if (mTItem.mElId)
      result.push_back(HcalElectronicsId(mTItem.mElId));

  return result;
}

std::vector<HcalElectronicsId> HcalElectronicsMap::allElectronicsIdPrecision() const {
  std::vector<HcalElectronicsId> result;
  for (auto mPItem : mPItems)
    if (mPItem.mElId)
      result.push_back(HcalElectronicsId(mPItem.mElId));
  return result;
}

std::vector<HcalElectronicsId> HcalElectronicsMap::allElectronicsIdTrigger() const {
  std::vector<HcalElectronicsId> result;
  for (auto mTItem : mTItems)
    if (mTItem.mElId)
      result.push_back(HcalElectronicsId(mTItem.mElId));

  return result;
}

std::vector<HcalGenericDetId> HcalElectronicsMap::allPrecisionId() const {
  std::vector<HcalGenericDetId> result;
  std::set<unsigned long> allIds;
  for (auto mPItem : mPItems)
    if (mPItem.mId)
      allIds.insert(mPItem.mId);
  for (unsigned long allId : allIds) {
    result.push_back(HcalGenericDetId(allId));
  }
  return result;
}

std::vector<HcalTrigTowerDetId> HcalElectronicsMap::allTriggerId() const {
  std::vector<HcalTrigTowerDetId> result;
  std::set<unsigned long> allIds;
  for (auto mTItem : mTItems)
    if (mTItem.mTrigId)
      allIds.insert(mTItem.mTrigId);
  for (unsigned long allId : allIds)
    result.push_back(HcalTrigTowerDetId(allId));
  return result;
}

//use helper to do mapping
HcalElectronicsMapAddons::Helper::Helper()
    : mPItems(HcalElectronicsId::maxLinearIndex + 1), mTItems(HcalElectronicsId::maxLinearIndex + 1) {}

bool HcalElectronicsMapAddons::Helper::mapEId2tId(HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  HcalElectronicsMap::TriggerItem& item = mTItems[fElectronicsId.linearIndex()];

  if (item.mElId == 0)
    item.mElId = fElectronicsId.rawId();
  if (item.mTrigId == 0) {
    item.mTrigId = fTriggerId.rawId();  // just cast avoiding long machinery
  } else if (item.mTrigId != fTriggerId.rawId()) {
    edm::LogWarning("HCAL") << "HcalElectronicsMap::Helper::mapEId2tId-> Electronics channel " << fElectronicsId
                            << " already mapped to trigger channel " << (HcalTrigTowerDetId(item.mTrigId))
                            << ". New value " << fTriggerId << " is ignored";
    return false;
  }
  return true;
}

bool HcalElectronicsMapAddons::Helper::mapEId2chId(HcalElectronicsId fElectronicsId, DetId fId) {
  HcalElectronicsMap::PrecisionItem& item = mPItems[fElectronicsId.linearIndex()];

  if (item.mElId == 0)
    item.mElId = fElectronicsId.rawId();
  if (item.mId == 0) {
    item.mId = fId.rawId();
  } else if (item.mId != fId.rawId()) {
    edm::LogWarning("HCAL") << "HcalElectronicsMap::Helper::mapEId2tId-> Electronics channel " << fElectronicsId
                            << " already mapped to channel " << HcalGenericDetId(item.mId) << ". New value "
                            << HcalGenericDetId(fId) << " is ignored";
    return false;
  }
  return true;
}

void HcalElectronicsMap::sortById() {
  HcalObjectAddons::sortByT<PrecisionItem, HcalElectronicsMapAddons::LessById>(mPItems, mPItemsById);
}

void HcalElectronicsMap::sortByTriggerId() {
  HcalObjectAddons::sortByT<TriggerItem, HcalElectronicsMapAddons::LessByTrigId>(mTItems, mTItemsByTrigId);
}

void HcalElectronicsMap::initialize() {
  sortById();
  sortByTriggerId();
}
