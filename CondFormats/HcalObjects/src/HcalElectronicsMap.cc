/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Hcal channels
$Author: ratnikov
$Date: 2007/02/19 23:33:42 $
$Revision: 1.17 $
*/

#include <iostream>
#include <set>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

HcalElectronicsMap::HcalElectronicsMap() 
{}

namespace hcal_impl {
  static const int PartialEIdMask = 0x3fff;
    class LessById {public: bool operator () (const HcalElectronicsMap::Item* a, const HcalElectronicsMap::Item* b) {return a->mId < b->mId;}};
    class LessByPartialEId {
    public: 
      bool operator () (const HcalElectronicsMap::Item* a, const HcalElectronicsMap::Item* b) {
	return (a->mElId&PartialEIdMask) < (b->mElId&PartialEIdMask);
      }
    };
    class LessByElId {public: 
      bool operator () (const HcalElectronicsMap::Item* a, const HcalElectronicsMap::Item* b) {return a->mElId < b->mElId;}
      bool operator () (const HcalElectronicsMap::Item& a, const HcalElectronicsMap::Item& b) {return a.mElId < b.mElId;}
    };
    class LessByTrigId {public: bool operator () (const HcalElectronicsMap::Item* a, const HcalElectronicsMap::Item* b) {return a->mTrigId < b->mTrigId;}};
}

HcalElectronicsMap::~HcalElectronicsMap(){}

const HcalElectronicsMap::Item* HcalElectronicsMap::findById (unsigned long fId) const {
  Item target (fId, 0, 0);
  std::vector<const HcalElectronicsMap::Item*>::const_iterator item;

  if (mItemsById.size()!=mItems.size()) sortById();
  
  item = std::lower_bound (mItemsById.begin(), mItemsById.end(), &target, hcal_impl::LessById());
  if (item == mItemsById.end() || (*item)->mId != fId) 
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const HcalElectronicsMap::Item* HcalElectronicsMap::findByPartialEId (unsigned long fElId) const {
  Item target (0, fElId, 0);
  std::vector<const HcalElectronicsMap::Item*>::const_iterator item;

  if (mItemsByPartialEId.size()!=mItems.size()) sortByPartialEId();
  
  item = std::lower_bound (mItemsByPartialEId.begin(), mItemsByPartialEId.end(), &target, hcal_impl::LessByPartialEId());
  if (item == mItemsByPartialEId.end() || ((*item)->mElId&hcal_impl::PartialEIdMask) != (fElId&hcal_impl::PartialEIdMask))
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const HcalElectronicsMap::Item* HcalElectronicsMap::findByElId (unsigned long fElId) const {
  Item target (0, fElId, 0);
  std::vector<HcalElectronicsMap::Item>::const_iterator item;

  item = std::lower_bound (mItems.begin(), mItems.end(), target, hcal_impl::LessByElId ());
  if (item == mItems.end() || item->mElId != fElId)
    // throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for e-cell " << fElId;
    return 0;
  return &*item;
}

const HcalElectronicsMap::Item* HcalElectronicsMap::findByTrigId (unsigned long fTrigId) const {
  Item target (0, 0, fTrigId);
  std::vector<const HcalElectronicsMap::Item*>::const_iterator item;

  if (mItemsByTrigId.size()!=mItems.size()) sortByTriggerId();
  
  item = std::lower_bound (mItemsByTrigId.begin(), mItemsByTrigId.end(), &target, hcal_impl::LessByTrigId());
  if (item == mItemsByTrigId.end() || (*item)->mTrigId != fTrigId) 
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const DetId HcalElectronicsMap::lookup(HcalElectronicsId fId ) const {
  const Item* item = findByElId (fId.rawId ());
  return DetId (item ? item->mId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookup(DetId fId) const {
  const Item* item = findById (fId.rawId ());
  return HcalElectronicsId (item ? item->mElId : 0);
}

const DetId HcalElectronicsMap::lookupTrigger(HcalElectronicsId fId) const {
  const Item* item = findByElId (fId.rawId ());
  return DetId (item ? item->mTrigId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookupTrigger(DetId fId) const {
  const Item* item = findByTrigId (fId.rawId ());
  return HcalElectronicsId (item ? item->mElId : 0);
}

bool HcalElectronicsMap::lookup(const HcalElectronicsId pId, HcalElectronicsId& eid, DetId& did) const {
  const Item* item = findByPartialEId (pId.rawId ());
  if (item!=0) {
    eid=HcalElectronicsId(item->mElId);
    did=DetId(item->mId);
  }
  return (item!=0);
}

bool HcalElectronicsMap::setMapping (DetId fId, HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  Item item (fId.rawId (), fElectronicsId.rawId (), fTriggerId.rawId ());
  mItems.push_back (item);
  return true;
}

std::vector <HcalElectronicsId> HcalElectronicsMap::allElectronicsId () const {
  std::vector <HcalElectronicsId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) 
    if (item->mElId) allIds.insert (item->mElId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++)
    result.push_back (HcalElectronicsId (*channel));
  return result;
}

std::vector <DetId> HcalElectronicsMap::allDetectorId () const {
  std::vector <DetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mId) allIds.insert (item->mId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++) {
      result.push_back (DetId (DetId (*channel)));
  }
  return result;
}

std::vector <HcalCalibDetId> HcalElectronicsMap::allCalibrationId () const {
  std::vector <HcalCalibDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mId) allIds.insert (item->mId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++) {
    if (HcalGenericDetId (*channel).isHcalCalibDetId ()) {
      result.push_back (HcalCalibDetId (*channel));
    }
  }
  return result;
}

std::vector <HcalDetId> HcalElectronicsMap::allHcalDetectorId () const {
  std::vector <HcalDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mId) allIds.insert (item->mId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++) {
    if (HcalGenericDetId (*channel).isHcalDetId()) {
      result.push_back (HcalDetId (*channel));
    }
  }
  return result;
}

std::vector <HcalTrigTowerDetId> HcalElectronicsMap::allTriggerId () const {
  std::vector <HcalTrigTowerDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mTrigId) allIds.insert (item->mTrigId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++)
    result.push_back (HcalTrigTowerDetId (*channel));
  return result;
}

bool HcalElectronicsMap::mapEId2tId (HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  const Item* item = findByElId (fElectronicsId.rawId ());
  if (item) { // record exists
    if (item->mTrigId == 0) {
      ((Item*)item)->mTrigId = fTriggerId.rawId (); // just cast avoiding long machinery
    } 
    else if (item->mTrigId != fTriggerId.rawId ()) {
      std::cerr << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId.rawId () << " already mapped to trigger channel " 
		<< item->mTrigId << ". New value " << fTriggerId.rawId () << " is ignored" << std::endl;
      return false;
    }
    return true;
  }
  else {
    bool retval=setMapping (DetId (), fElectronicsId, fTriggerId);
    // resort (almost certainly needed)
    sortByElectronicsId();
    return retval;
  }
}

bool HcalElectronicsMap::mapEId2chId (HcalElectronicsId fElectronicsId, DetId fId) {
  const Item* item = findByElId (fElectronicsId.rawId ());
  if (item) { // record exists
    if (item->mId == 0) {
      ((Item*)item)->mId = fId.rawId (); // just cast avoiding long machinery
    } 
    else if (item->mId != fId.rawId ()) {
      std::cerr << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId.rawId () << " already mapped to channel " 
		<< item->mId << ". New value " << fId.rawId () << " is ignored" << std::endl;
      return false;
    }
    return true;
  }
  bool retval=setMapping (fId, fElectronicsId, 0);
  // resort (almost certainly needed)
  sortByElectronicsId();
  return retval;
}

void HcalElectronicsMap::sortById () const {
  if (mItems.size()!=mItemsById.size()) {
    mItemsById.clear();
    mItemsById.reserve(mItems.size());
    for (std::vector<Item>::const_iterator i=mItems.begin(); i!=mItems.end(); ++i) {
      mItemsById.push_back(&(*i));
    }
  }
  std::sort (mItemsById.begin(), mItemsById.end(), hcal_impl::LessById ());
}

void HcalElectronicsMap::sortByPartialEId () const {
  if (mItems.size()!=mItemsByPartialEId.size()) {
    mItemsByPartialEId.clear();
    mItemsByPartialEId.reserve(mItems.size());
    for (std::vector<Item>::const_iterator i=mItems.begin(); i!=mItems.end(); ++i) {
      mItemsByPartialEId.push_back(&(*i));
    }
  }
  std::sort (mItemsByPartialEId.begin(), mItemsByPartialEId.end(), hcal_impl::LessByPartialEId());
}
void HcalElectronicsMap::sortByElectronicsId () {
  std::sort (mItems.begin(), mItems.end(), hcal_impl::LessByElId ());
}
void HcalElectronicsMap::sortByTriggerId () const {
  if (mItems.size()!=mItemsByTrigId.size()) {
    mItemsByTrigId.clear();
    mItemsByTrigId.reserve(mItems.size());
    for (std::vector<Item>::const_iterator i=mItems.begin(); i!=mItems.end(); ++i) {
      mItemsByTrigId.push_back(&(*i));
    }
  }
  std::sort (mItemsByTrigId.begin(), mItemsByTrigId.end(), hcal_impl::LessByTrigId ());
}
