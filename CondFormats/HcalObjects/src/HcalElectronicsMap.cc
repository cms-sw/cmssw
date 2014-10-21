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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalElectronicsMap::HcalElectronicsMap() : 
  mPItems(HcalElectronicsId::maxLinearIndex+1),
  mTItems(HcalElectronicsId::maxLinearIndex+1),
  mPItemsById(nullptr), mTItemsByTrigId(nullptr)
{}

namespace hcal_impl {
  class LessById {public: bool operator () (const HcalElectronicsMap::PrecisionItem* a, const HcalElectronicsMap::PrecisionItem* b) {return a->mId < b->mId;}};
  class LessByTrigId {public: bool operator () (const HcalElectronicsMap::TriggerItem* a, const HcalElectronicsMap::TriggerItem* b) {return a->mTrigId < b->mTrigId;}};
}

HcalElectronicsMap::~HcalElectronicsMap() {
    delete mPItemsById.load();
    delete mTItemsByTrigId.load();
}
// copy-ctor
HcalElectronicsMap::HcalElectronicsMap(const HcalElectronicsMap& src)
    : mPItems(src.mPItems), mTItems(src.mTItems),
      mPItemsById(nullptr), mTItemsByTrigId(nullptr) {}
// copy assignment operator
HcalElectronicsMap&
HcalElectronicsMap::operator=(const HcalElectronicsMap& rhs) {
    HcalElectronicsMap temp(rhs);
    temp.swap(*this);
    return *this;
}
// public swap function
void HcalElectronicsMap::swap(HcalElectronicsMap& other) {
    std::swap(mPItems, other.mPItems);
    std::swap(mTItems, other.mTItems);
    other.mTItemsByTrigId.exchange(
            mTItemsByTrigId.exchange(other.mTItemsByTrigId.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
    other.mPItemsById.exchange(
            mPItemsById.exchange(other.mPItemsById.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
// move constructor
HcalElectronicsMap::HcalElectronicsMap(HcalElectronicsMap&& other) 
    : HcalElectronicsMap() {
    other.swap(*this);
}

const HcalElectronicsMap::PrecisionItem* HcalElectronicsMap::findById (unsigned long fId) const {
  PrecisionItem target (fId, 0);
  std::vector<const HcalElectronicsMap::PrecisionItem*>::const_iterator item;

  sortById();
  
  auto ptr = (*mPItemsById.load(std::memory_order_acquire));
  item = std::lower_bound (ptr.begin(), ptr.end(), &target, hcal_impl::LessById());
  if (item == ptr.end() || (*item)->mId != fId)
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const HcalElectronicsMap::PrecisionItem* HcalElectronicsMap::findPByElId (unsigned long fElId) const {
  HcalElectronicsId eid(fElId);
  const PrecisionItem* i=&(mPItems[eid.linearIndex()]);
  
  if (i!=0 && i->mElId!=fElId) i=0;
  return i;
}

const HcalElectronicsMap::TriggerItem* HcalElectronicsMap::findTByElId (unsigned long fElId) const {
  HcalElectronicsId eid(fElId);
  const TriggerItem* i=&(mTItems[eid.linearIndex()]);
  
  if (i!=0 && i->mElId!=fElId) i=0;
  return i;
}


const HcalElectronicsMap::TriggerItem* HcalElectronicsMap::findByTrigId (unsigned long fTrigId) const {
  TriggerItem target (fTrigId,0);
  std::vector<const HcalElectronicsMap::TriggerItem*>::const_iterator item;

  sortByTriggerId();
  
  auto ptr = (*mTItemsByTrigId.load(std::memory_order_acquire));
  item = std::lower_bound (ptr.begin(), ptr.end(), &target, hcal_impl::LessByTrigId());
  if (item == (*mTItemsByTrigId).end() || (*item)->mTrigId != fTrigId)
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const DetId HcalElectronicsMap::lookup(HcalElectronicsId fId ) const {
  const PrecisionItem* item = findPByElId (fId.rawId ());
  return DetId (item ? item->mId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookup(DetId fId) const {
  const PrecisionItem* item = findById (fId.rawId ());
  return HcalElectronicsId (item ? item->mElId : 0);
}

const DetId HcalElectronicsMap::lookupTrigger(HcalElectronicsId fId) const {
  const TriggerItem* item = findTByElId (fId.rawId ());
  return DetId (item ? item->mTrigId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookupTrigger(DetId fId) const {
  const TriggerItem* item = findByTrigId (fId.rawId ());
  return HcalElectronicsId (item ? item->mElId : 0);
}

bool HcalElectronicsMap::lookup(const HcalElectronicsId pid, HcalElectronicsId& eid, HcalGenericDetId& did) const {
  const PrecisionItem* i=&(mPItems[pid.linearIndex()]);
  if (i!=0 && i->mId!=0) {
    eid=HcalElectronicsId(i->mElId);
    did=HcalGenericDetId(i->mId);
    return true;
  } else return false;
}

bool HcalElectronicsMap::lookup(const HcalElectronicsId pid, HcalElectronicsId& eid, HcalTrigTowerDetId& did) const {
  const TriggerItem* i=&(mTItems[pid.linearIndex()]);
  if (i!=0 && i->mTrigId!=0) {
    eid=HcalElectronicsId(i->mElId);
    did=HcalGenericDetId(i->mTrigId);
    return true;
  } else return false;  
}


std::vector <HcalElectronicsId> HcalElectronicsMap::allElectronicsId () const {
  std::vector <HcalElectronicsId> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); item != mPItems.end (); item++) 
    if (item->mElId) result.push_back(HcalElectronicsId(item->mElId));
  for (std::vector<TriggerItem>::const_iterator item = mTItems.begin (); item != mTItems.end (); item++) 
    if (item->mElId) result.push_back(HcalElectronicsId(item->mElId));

  return result;
}

std::vector <HcalElectronicsId> HcalElectronicsMap::allElectronicsIdPrecision() const {
  std::vector <HcalElectronicsId> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); item != mPItems.end (); item++) 
    if (item->mElId) result.push_back(HcalElectronicsId(item->mElId));
  return result;
}

std::vector <HcalElectronicsId> HcalElectronicsMap::allElectronicsIdTrigger() const {
  std::vector <HcalElectronicsId> result;
  for (std::vector<TriggerItem>::const_iterator item = mTItems.begin (); item != mTItems.end (); item++) 
    if (item->mElId) result.push_back(HcalElectronicsId(item->mElId));

  return result;
}

std::vector <HcalGenericDetId> HcalElectronicsMap::allPrecisionId () const {
  std::vector <HcalGenericDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); item != mPItems.end (); item++)  
    if (item->mId) allIds.insert (item->mId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++) {
      result.push_back (HcalGenericDetId (*channel));
  }
  return result;
}

std::vector <HcalTrigTowerDetId> HcalElectronicsMap::allTriggerId () const {
  std::vector <HcalTrigTowerDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<TriggerItem>::const_iterator item = mTItems.begin (); item != mTItems.end (); item++)  
    if (item->mTrigId) allIds.insert (item->mTrigId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++)
    result.push_back (HcalTrigTowerDetId (*channel));
  return result;
}

bool HcalElectronicsMap::mapEId2tId (HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  TriggerItem& item = mTItems[fElectronicsId.linearIndex()];

  if (mTItemsByTrigId) {
      delete mTItemsByTrigId.load();
      mTItemsByTrigId = nullptr;
  }

  if (item.mElId==0) item.mElId=fElectronicsId.rawId();
  if (item.mTrigId == 0) {
    item.mTrigId = fTriggerId.rawId (); // just cast avoiding long machinery
  } 
  else if (item.mTrigId != fTriggerId.rawId ()) {
    edm::LogWarning("HCAL") << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId  << " already mapped to trigger channel " 
	      << (HcalTrigTowerDetId(item.mTrigId)) << ". New value " << fTriggerId << " is ignored" ;
    return false;
  }
  return true;
}

bool HcalElectronicsMap::mapEId2chId (HcalElectronicsId fElectronicsId, DetId fId) {
  PrecisionItem& item = mPItems[fElectronicsId.linearIndex()];

  delete mPItemsById.load();
  mPItemsById = nullptr;

  if (item.mElId==0) item.mElId=fElectronicsId.rawId();
  if (item.mId == 0) {
    item.mId = fId.rawId ();
  } 
  else if (item.mId != fId.rawId ()) {
     edm::LogWarning("HCAL") << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId << " already mapped to channel " 
			     << HcalGenericDetId(item.mId) << ". New value " << HcalGenericDetId(fId) << " is ignored" ;
       return false;
  }
  return true;
}

void HcalElectronicsMap::sortById () const {
  if (!mPItemsById.load(std::memory_order_acquire)) {
      auto ptr = new std::vector<const PrecisionItem*>;
      for (auto i=mPItems.begin(); i!=mPItems.end(); ++i) {
          if (i->mElId) (*ptr).push_back(&(*i));
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

void HcalElectronicsMap::sortByTriggerId () const {
  if (!mTItemsByTrigId.load(std::memory_order_acquire)) {
      auto ptr = new std::vector<const TriggerItem*>;
      for (auto i=mTItems.begin(); i!=mTItems.end(); ++i) {
          if (i->mElId) (*ptr).push_back(&(*i));
      }
    
      std::sort ((*ptr).begin(), (*ptr).end(), hcal_impl::LessByTrigId ());
      //atomically try to swap this to become mTItemsByTrigId
      std::vector<const TriggerItem*>* expect = nullptr;
      bool exchanged = mTItemsByTrigId.compare_exchange_strong(expect, ptr, std::memory_order_acq_rel);
      if(!exchanged) {
          delete ptr;
      }
  }
}
