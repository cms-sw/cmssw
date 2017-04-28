#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalFrontEndMap::HcalFrontEndMap(const HcalFrontEndMapAddons::Helper& helper) :
    mPItems(helper.mPItems.begin(),helper.mPItems.end())
{
  initialize();
}


HcalFrontEndMap::~HcalFrontEndMap() {
}

// copy-ctor
HcalFrontEndMap::HcalFrontEndMap(const HcalFrontEndMap& src)
    : mPItems(src.mPItems), mPItemsById(src.mPItemsById) {}

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
HcalFrontEndMap::HcalFrontEndMap(HcalFrontEndMap&& other) : HcalFrontEndMap() {
  other.swap(*this);
}

const HcalFrontEndMap::PrecisionItem* HcalFrontEndMap::findById (uint32_t fId, const std::vector<const PrecisionItem*>& mPItemsById) {
  PrecisionItem target (fId, 0, "");

  auto item = std::lower_bound (mPItemsById.begin(), mPItemsById.end(), &target, HcalFrontEndMapAddons::LessById());
  if (item == mPItemsById.end() || (*item)->mId != fId)
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

HcalFrontEndMapAddons::Helper::Helper()
{
}

bool HcalFrontEndMapAddons::Helper::loadObject(DetId fId, int rm, std::string rbx ) {
  HcalFrontEndMap::PrecisionItem target (fId.rawId(), rm, rbx);
  auto iter = mPItems.find(target);
  if (iter!=mPItems.end()) {
    edm::LogWarning("HCAL") << "HcalFrontEndMap::loadObject DetId " 
			    << HcalDetId(fId) << " already exists with RM "
			    << iter->mRM << " RBX " << iter->mRBX 
			    << " new values " << rm << " and " << rbx
			    << " are ignored";
    return false;
  } else {
    mPItems.insert(target);
    return true;
  }
}

const int HcalFrontEndMap::lookupRM(DetId fId) const {
  const PrecisionItem* item = HcalFrontEndMap::findById (fId.rawId (),mPItemsById);
  return (item ? item->mRM : 0);
}

const int HcalFrontEndMap::lookupRMIndex(DetId fId) const {
  const PrecisionItem* item = HcalFrontEndMap::findById (fId.rawId (),mPItemsById);
  HcalFrontEndId id;
  if (item) id = HcalFrontEndId(item->mRBX,item->mRM,0,1,0,1,0);
  return id.rmIndex();
}

const std::string HcalFrontEndMap::lookupRBX(DetId fId) const {
  const PrecisionItem* item = HcalFrontEndMap::findById (fId.rawId (),mPItemsById);
  return (item ? item->mRBX : "");
}

const int HcalFrontEndMap::lookupRBXIndex(DetId fId) const {
  const PrecisionItem* item = HcalFrontEndMap::findById (fId.rawId (),mPItemsById);
  HcalFrontEndId id;
  if (item) id = HcalFrontEndId(item->mRBX,item->mRM,0,1,0,1,0);
  return id.rbxIndex();
}

std::vector <DetId> HcalFrontEndMap::allDetIds() const {
  std::vector <DetId> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); 
       item != mPItems.end (); item++) 
    if (item->mId) result.push_back(DetId(item->mId));
  return result;
}

std::vector <int> HcalFrontEndMap::allRMs () const {
  std::vector <int> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin ();
       item != mPItems.end (); item++) {
    if (std::find(result.begin(),result.end(),item->mRM) == result.end())
      result.push_back(item->mRM);
  }
  return result;
}

std::vector <std::string> HcalFrontEndMap::allRBXs() const {
  std::vector <std::string> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); 
       item != mPItems.end (); item++)  {
    if (std::find(result.begin(),result.end(),item->mRBX) == result.end())
      result.push_back(item->mRBX);
  }
  return result;
}

void HcalFrontEndMap::sortById (const std::vector<PrecisionItem>& items, std::vector<const PrecisionItem*>& itemsById) {
  itemsById.clear();
  itemsById.reserve(items.size());
  for(const auto& i : items){
    if(i.mId) itemsById.push_back(&i);
  }
  std::sort (itemsById.begin(), itemsById.end(), HcalFrontEndMapAddons::LessById ());
}

void HcalFrontEndMap::initialize() {
  HcalFrontEndMap::sortById(mPItems,mPItemsById);
 }
