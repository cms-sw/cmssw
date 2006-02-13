/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Hcal channels
$Author: ratnikov
$Date: 2005/12/27 23:50:28 $
$Revision: 1.4 $
*/

#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

HcalElectronicsMap::HcalElectronicsMap() 
  : mSortedByChId (false),
    mSortedByElId (false),
    mSortedByTrigId (false)
{}

HcalElectronicsMap::~HcalElectronicsMap(){}

const HcalElectronicsMap::Item* HcalElectronicsMap::findByChId (unsigned long fChId, bool fWarning) const {
  Item target (fChId, 0, 0);
  std::vector<HcalElectronicsMap::Item>::const_iterator item;
  if (mSortedByChId) {
    item = std::lower_bound (mItems.begin(), mItems.end(), target, Item::LessByChId ());
  }
  else {
    if (fWarning) std::cerr << "HcalElectronicsMap-> container is not sorted. Use sortByChaId () to search effectively" << std::endl;
    item = mItems.begin();
    Item::LessByChId less;
    while (item != mItems.end() && !( less (*item,target) || less (target, *item))) item++;
  }
  if (item == mItems.end() || item->mChId != fChId) return 0;
  return &*item;
}

const HcalElectronicsMap::Item* HcalElectronicsMap::findByElId (unsigned long fElId, bool fWarning) const {
  Item target (0, fElId, 0);
  std::vector<HcalElectronicsMap::Item>::const_iterator item;
  if (mSortedByElId) {
    item = std::lower_bound (mItems.begin(), mItems.end(), target, Item::LessByElId ());
  }
  else {
    if (fWarning) std::cerr << "HcalElectronicsMap-> container is not sorted. Use sortById () to search effectively" << std::endl;
    item = mItems.begin();
    Item::LessByElId less;
    while (! (item == mItems.end() || ( !less (*item,target) && !less (target, *item)))) item++;
  }
  if (item == mItems.end() || item->mElId != fElId) return 0;
  return &*item;
}

const HcalElectronicsMap::Item* HcalElectronicsMap::findByTrigId (unsigned long fTrigId, bool fWarning) const {
  Item target (0, 0, fTrigId);
  std::vector<HcalElectronicsMap::Item>::const_iterator item;
  if (mSortedByTrigId) {
    item = std::lower_bound (mItems.begin(), mItems.end(), target, Item::LessByTrigId ());
  }
  else {
    if (fWarning) std::cerr << "HcalElectronicsMap-> container is not sorted. Use sortByTrigId () to search effectively" << std::endl;
    item = mItems.begin();
    Item::LessByTrigId less;
    while (item != mItems.end() && !( less (*item,target) || less (target, *item))) item++;
  }
  if (item == mItems.end() || item->mTrigId != fTrigId) return 0;
  return &*item;
}


const HcalDetId HcalElectronicsMap::lookup(HcalElectronicsId fId, bool fWarning ) const {
  const Item* item = findByElId (fId.rawId (), fWarning);
  return HcalDetId (item ? item->mChId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookup(HcalDetId fId, bool fWarning) const {
  const Item* item = findByChId (fId.rawId (), fWarning);
  return HcalElectronicsId (item ? item->mElId : 0);
}

const HcalTrigTowerDetId HcalElectronicsMap::lookupTrigger(HcalElectronicsId fId, bool fWarning) const {
  const Item* item = findByElId (fId.rawId (), fWarning);
  return HcalTrigTowerDetId (item ? item->mTrigId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookupTrigger(HcalTrigTowerDetId fId, bool fWarning) const {
  const Item* item = findByTrigId (fId.rawId (), fWarning);
  return HcalElectronicsId (item ? item->mElId : 0);
}

bool HcalElectronicsMap::known (HcalElectronicsId fId, bool fWarning ) const {
  return findByElId (fId.rawId (), fWarning);
}

bool HcalElectronicsMap::known (HcalDetId fId, bool fWarning ) const {
  return findByChId (fId.rawId (), fWarning);
}

bool HcalElectronicsMap::known (HcalTrigTowerDetId fId, bool fWarning ) const {
  return findByTrigId (fId.rawId (), fWarning);
}

bool HcalElectronicsMap::setMapping (HcalDetId fChId, HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  Item item (fChId.rawId (), fElectronicsId.rawId (), fTriggerId.rawId ());
  mItems.push_back (item);
  mSortedByChId = mSortedByElId = mSortedByTrigId = false;
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

std::vector <HcalDetId> HcalElectronicsMap::allDetectorId () const {
  std::vector <HcalDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mChId) allIds.insert (item->mChId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++)
    result.push_back (HcalDetId (*channel));
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
  const Item* item = findByElId (fElectronicsId.rawId (), false);
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
    return setMapping (0, fElectronicsId, fTriggerId);
  }
}

bool HcalElectronicsMap::mapEId2chId (HcalElectronicsId fElectronicsId, HcalDetId fChId) {
  const Item* item = findByElId (fElectronicsId.rawId (), false);
  if (item) { // record exists
    if (item->mChId == 0) {
      ((Item*)item)->mChId = fChId.rawId (); // just cast avoiding long machinery
    } 
    else if (item->mChId != fChId.rawId ()) {
      std::cerr << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId.rawId () << " already mapped to channel " 
		<< item->mChId << ". New value " << fChId.rawId () << " is ignored" << std::endl;
      return false;
    }
    return true;
  }
  return setMapping (fChId, fElectronicsId, 0);
}

void HcalElectronicsMap::sortByChaId () {
    if (!mSortedByChId) {
    std::sort (mItems.begin(), mItems.end(), Item::LessByChId ());
    mSortedByChId = true;
    mSortedByElId = mSortedByTrigId = false;
  }

}
void HcalElectronicsMap::sortByElectronicsId () {
    if (!mSortedByElId) {
    std::sort (mItems.begin(), mItems.end(), Item::LessByElId ());
    mSortedByElId = true;
    mSortedByChId = mSortedByTrigId = false;
  }

}
void HcalElectronicsMap::sortByTriggerId () {
    if (!mSortedByTrigId) {
    std::sort (mItems.begin(), mItems.end(), Item::LessByTrigId ());
    mSortedByTrigId = true;
    mSortedByChId = mSortedByElId = false;
  }

}
