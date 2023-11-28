/** 
\class HcalDcsMap
\author Gena Kukartsev
POOL object to store map between detector ID and DCS ID
Inspired by HcalElectronicsMap
$Author: kukartse
$Date: 2010/02/22 21:08:07 $
$Revision: 1.1 $
*/

#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalDcsMap.h"
#include "CondFormats/HcalObjects/interface/HcalObjectAddons.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalDcsMap::HcalDcsMap(const HcalDcsMapAddons::Helper& helper) : mItems(helper.mItems.begin(), helper.mItems.end()) {
  initialize();
}

HcalDcsMap::~HcalDcsMap() {}
// copy-ctor
HcalDcsMap::HcalDcsMap(const HcalDcsMap& src)
    : mItems(src.mItems), mItemsById(src.mItemsById), mItemsByDcsId(src.mItemsByDcsId) {}
// copy assignment operator
HcalDcsMap& HcalDcsMap::operator=(const HcalDcsMap& rhs) {
  HcalDcsMap temp(rhs);
  temp.swap(*this);
  return *this;
}
// public swap function
void HcalDcsMap::swap(HcalDcsMap& other) {
  std::swap(mItems, other.mItems);
  std::swap(mItemsById, other.mItemsById);
  std::swap(mItemsByDcsId, other.mItemsByDcsId);
}
// move constructor
HcalDcsMap::HcalDcsMap(HcalDcsMap&& other) : HcalDcsMap() { other.swap(*this); }

HcalDcsMap::const_iterator HcalDcsMap::beginById(void) const {
  const_iterator _iter;
  _iter.fIter = mItemsById.begin();
  return _iter;
}

HcalDcsMap::const_iterator HcalDcsMap::beginByDcsId(void) const {
  const_iterator _iter;
  _iter.fIter = mItemsByDcsId.begin();
  return _iter;
}

HcalDcsMap::const_iterator HcalDcsMap::endById(void) const {
  const_iterator _iter;
  _iter.fIter = mItemsById.end();
  return _iter;
}

HcalDcsMap::const_iterator HcalDcsMap::endByDcsId(void) const {
  const_iterator _iter;
  _iter.fIter = mItemsByDcsId.end();
  return _iter;
}

// iterator methods
bool HcalDcsMap::const_iterator::operator!=(const HcalDcsMap::const_iterator& other) {
  if (fIter != other.fIter)
    return true;
  else
    return false;
}

HcalDcsMap::const_iterator HcalDcsMap::const_iterator::operator++() {
  ++fIter;
  return *this;
}

HcalDcsMap::const_iterator HcalDcsMap::const_iterator::operator++(int) {
  const_iterator i = *this;
  ++fIter;
  return i;
}

void HcalDcsMap::const_iterator::next(void) { ++fIter; }

HcalDcsDetId HcalDcsMap::const_iterator::getHcalDcsDetId(void) { return (*fIter)->mDcsId; }

HcalDetId HcalDcsMap::const_iterator::getHcalDetId(void) { return (*fIter)->mId; }

const HcalDcsMap::Item* HcalDcsMap::findById(unsigned long fId) const {
  Item target(fId, 0);
  return HcalObjectAddons::findByT<Item, HcalDcsMapAddons::LessById>(&target, mItemsById);
}

const HcalDcsMap::Item* HcalDcsMap::findByDcsId(unsigned long fDcsId) const {
  Item target(0, fDcsId);
  return HcalObjectAddons::findByT<Item, HcalDcsMapAddons::LessByDcsId>(&target, mItemsByDcsId);
}

HcalDetId HcalDcsMap::lookup(HcalDcsDetId fId) const {
  // DCS type is a part of DcsDetId but it does not make sense to keep
  // duplicate records in the map for DCS channels where only type is different.
  // Hence, the type in HcalDcsDetId is always forced to DCSUNKNOWN
  HcalDcsDetId fDcsId_notype(fId.subdet(),
                             fId.ring(),  // side is already included
                             fId.slice(),
                             HcalDcsDetId::DCSUNKNOWN,
                             fId.subchannel());
  auto item = HcalDcsMap::findByDcsId(fDcsId_notype.rawId());
  return item ? item->mId : 0;
}

HcalDcsDetId HcalDcsMap::lookup(HcalDetId fId, HcalDcsDetId::DcsType type) const {
  auto item = HcalDcsMap::findById(fId.rawId());
  HcalDcsDetId _id(item ? item->mId : 0);
  return HcalDcsDetId(_id.subdet(), _id.zside() * _id.ring(), _id.slice(), type, _id.subchannel());
}

//FIXME: remove duplicates
std::vector<HcalDcsDetId> HcalDcsMap::allHcalDcsDetId() const {
  std::vector<HcalDcsDetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin(); item != mItems.end(); item++)
    if (item->mDcsId)
      result.push_back(HcalDcsDetId(item->mDcsId));
  return result;
}

// FIXME: remove duplicates
std::vector<HcalGenericDetId> HcalDcsMap::allHcalDetId() const {
  std::vector<HcalGenericDetId> result;
  std::set<unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin(); item != mItems.end(); item++)
    if (item->mId)
      allIds.insert(item->mId);
  for (std::set<unsigned long>::const_iterator channel = allIds.begin(); channel != allIds.end(); channel++) {
    result.push_back(HcalGenericDetId(*channel));
  }
  return result;
}

HcalDcsMapAddons::Helper::Helper() {}

bool HcalDcsMapAddons::Helper::mapGeomId2DcsId(HcalDetId fId, HcalDcsDetId fDcsId) {
  // DCS type is a part of DcsDetId but it does not make sense to keep
  // duplicate records in the map for DCS channels where only type is different.
  // Hence, the type in HcalDcsDetId is always forced to DCSUNKNOWN
  HcalDcsDetId fDcsId_notype(fDcsId.subdet(),
                             fDcsId.ring(),  // side is included
                             fDcsId.slice(),
                             HcalDcsDetId::DCSUNKNOWN,
                             fDcsId.subchannel());
  HcalDcsMap::Item target(fId, fDcsId_notype);
  auto iter = mItems.find(target);
  if (iter != mItems.end() and iter->mId == static_cast<uint32_t>(fId)) {
    edm::LogWarning("HCAL") << "HcalDcsMap::mapGeomId2DcsId-> Geom channel " << fId << " already mapped to DCS channel "
                            << fDcsId_notype;
    return false;  // element already exists
  }
  mItems.insert(target);

  return true;
}

void HcalDcsMap::sortById() { HcalObjectAddons::sortByT<Item, HcalDcsMapAddons::LessById>(mItems, mItemsById); }
void HcalDcsMap::sortByDcsId() {
  HcalObjectAddons::sortByT<Item, HcalDcsMapAddons::LessByDcsId>(mItems, mItemsByDcsId);
}

void HcalDcsMap::initialize() {
  sortById();
  sortByDcsId();
}
