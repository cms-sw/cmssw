#include <algorithm>
#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalFrontEndMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalFrontEndId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalFrontEndMap::HcalFrontEndMap() : mPItemsById(nullptr) {}

namespace hcal_impl {
  class LessById {public: bool operator () (const HcalFrontEndMap::PrecisionItem* a, const HcalFrontEndMap::PrecisionItem* b) {return a->mId < b->mId;}};
}

HcalFrontEndMap::~HcalFrontEndMap() {
    delete mPItemsById.load();
}

// copy-ctor
HcalFrontEndMap::HcalFrontEndMap(const HcalFrontEndMap& src)
    : mPItems(src.mPItems), mPItemsById(nullptr) {}

// copy assignment operator
HcalFrontEndMap& HcalFrontEndMap::operator=(const HcalFrontEndMap& rhs) {
    HcalFrontEndMap temp(rhs);
    temp.swap(*this);
    return *this;
}

// public swap function
void HcalFrontEndMap::swap(HcalFrontEndMap& other) {
  std::swap(mPItems, other.mPItems);
  other.mPItemsById.exchange(mPItemsById.exchange(other.mPItemsById.load(std::memory_order_acquire), 
						  std::memory_order_acq_rel),
			     std::memory_order_acq_rel);
}

// move constructor
HcalFrontEndMap::HcalFrontEndMap(HcalFrontEndMap&& other) : HcalFrontEndMap() {
  other.swap(*this);
}

const HcalFrontEndMap::PrecisionItem* HcalFrontEndMap::findById (uint32_t fId) const {
  PrecisionItem target (fId, 0, "");
  std::vector<const HcalFrontEndMap::PrecisionItem*>::const_iterator item;

  sortById();
  auto const& ptr = (*mPItemsById.load(std::memory_order_acquire));
  item = std::lower_bound (ptr.begin(), ptr.end(), &target, hcal_impl::LessById());
  if (item == ptr.end() || (*item)->mId != fId)
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

bool HcalFrontEndMap::loadObject(DetId fId, int rm, std::string rbx ) {
  const PrecisionItem* item = findById (fId.rawId ());
  if (item) {
    edm::LogWarning("HCAL") << "HcalFrontEndMap::loadObject DetId " 
			    << HcalDetId(fId) << " already exists with RM "
			    << item->mRM << " RBX " << item->mRBX 
			    << " new values " << rm << " and " << rbx
			    << " are ignored";
    return false;
  } else {
    PrecisionItem target (fId.rawId(), rm, rbx);
    mPItems.push_back(target);
    return true;
  }
}

const int HcalFrontEndMap::lookupRM(DetId fId) const {
  const PrecisionItem* item = findById (fId.rawId ());
  return (item ? item->mRM : 0);
}

const int HcalFrontEndMap::lookupRMIndex(DetId fId) const {
  const PrecisionItem* item = findById (fId.rawId ());
  HcalFrontEndId id;
  if (item) id = HcalFrontEndId(item->mRBX,item->mRM,0,1,0,1,0);
  return id.rmIndex();
}

const std::string HcalFrontEndMap::lookupRBX(DetId fId) const {
  const PrecisionItem* item = findById (fId.rawId ());
  return (item ? item->mRBX : "");
}

const int HcalFrontEndMap::lookupRBXIndex(DetId fId) const {
  const PrecisionItem* item = findById (fId.rawId ());
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

void HcalFrontEndMap::sortById () const {
  if (!mPItemsById.load(std::memory_order_acquire)) {
    auto ptr = new std::vector<const PrecisionItem*>;
    for (auto i=mPItems.begin(); i!=mPItems.end(); ++i) {
      if (i->mId) (*ptr).push_back(&(*i));
    }
    
    std::sort ((*ptr).begin(), (*ptr).end(), hcal_impl::LessById ());
    //atomically try to swap this to become mPItemsById
    std::vector<const PrecisionItem*>* expect = nullptr;
    bool exchanged = mPItemsById.compare_exchange_strong(expect, ptr, std::memory_order_acq_rel);
    if(!exchanged) {
      delete ptr;
    }
  }
}
