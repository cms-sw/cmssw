#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "CondFormats/HcalObjects/interface/HcalObjectAddons.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalFrontEndMap::HcalFrontEndMap(const HcalFrontEndMapAddons::Helper& helper)
    : mPItems(helper.mPItems.begin(), helper.mPItems.end()) {
  initialize();
}

HcalFrontEndMap::~HcalFrontEndMap() {}

// copy-ctor
HcalFrontEndMap::HcalFrontEndMap(const HcalFrontEndMap& src) : mPItems(src.mPItems), mPItemsById(src.mPItemsById) {}

// copy assignment operator
HcalFrontEndMap& HcalFrontEndMap::operator=(const HcalFrontEndMap& rhs) {
  HcalFrontEndMap temp(rhs);
  temp.swap(*this);
  return *this;
}

// public swap function
void HcalFrontEndMap::swap(HcalFrontEndMap& other) {
  std::swap(mPItems, other.mPItems);
  std::swap(mPItemsById, other.mPItemsById);
}

// move constructor
HcalFrontEndMap::HcalFrontEndMap(HcalFrontEndMap&& other) : HcalFrontEndMap() { other.swap(*this); }

const HcalFrontEndMap::PrecisionItem* HcalFrontEndMap::findById(uint32_t fId) const {
  PrecisionItem target(fId, 0, "");
  return HcalObjectAddons::findByT<PrecisionItem, HcalFrontEndMapAddons::LessById>(&target, mPItemsById);
}

HcalFrontEndMapAddons::Helper::Helper() {}

bool HcalFrontEndMapAddons::Helper::loadObject(DetId fId, int rm, std::string rbx) {
  HcalFrontEndMap::PrecisionItem target(fId.rawId(), rm, rbx);
  auto iter = mPItems.find(target);
  if (iter != mPItems.end()) {
    edm::LogWarning("HCAL") << "HcalFrontEndMap::loadObject DetId " << HcalDetId(fId) << " already exists with RM "
                            << iter->mRM << " RBX " << iter->mRBX << " new values " << rm << " and " << rbx
                            << " are ignored";
    return false;
  } else {
    mPItems.insert(target);
    return true;
  }
}

const int HcalFrontEndMap::lookupRM(DetId fId) const {
  const PrecisionItem* item = findById(fId.rawId());
  return (item ? item->mRM : 0);
}

const int HcalFrontEndMap::lookupRMIndex(DetId fId) const {
  const PrecisionItem* item = findById(fId.rawId());
  HcalFrontEndId id;
  if (item)
    id = HcalFrontEndId(item->mRBX, item->mRM, 0, 1, 0, 1, 0);
  return id.rmIndex();
}

const std::string HcalFrontEndMap::lookupRBX(DetId fId) const {
  const PrecisionItem* item = findById(fId.rawId());
  return (item ? item->mRBX : "");
}

const int HcalFrontEndMap::lookupRBXIndex(DetId fId) const {
  const PrecisionItem* item = findById(fId.rawId());
  HcalFrontEndId id;
  if (item)
    id = HcalFrontEndId(item->mRBX, item->mRM, 0, 1, 0, 1, 0);
  return id.rbxIndex();
}

std::vector<DetId> HcalFrontEndMap::allDetIds() const {
  std::vector<DetId> result;
  for (const auto& mPItem : mPItems)
    if (mPItem.mId)
      result.push_back(DetId(mPItem.mId));
  return result;
}

std::vector<int> HcalFrontEndMap::allRMs() const {
  std::vector<int> result;
  for (const auto& mPItem : mPItems) {
    if (std::find(result.begin(), result.end(), mPItem.mRM) == result.end())
      result.push_back(mPItem.mRM);
  }
  return result;
}

std::vector<std::string> HcalFrontEndMap::allRBXs() const {
  std::vector<std::string> result;
  for (const auto& mPItem : mPItems) {
    if (std::find(result.begin(), result.end(), mPItem.mRBX) == result.end())
      result.push_back(mPItem.mRBX);
  }
  return result;
}

void HcalFrontEndMap::sortById() {
  HcalObjectAddons::sortByT<PrecisionItem, HcalFrontEndMapAddons::LessById>(mPItems, mPItemsById);
}

void HcalFrontEndMap::initialize() { HcalFrontEndMap::sortById(); }
