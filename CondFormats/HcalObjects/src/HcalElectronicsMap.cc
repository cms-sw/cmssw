/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Hcal channels
$Author: ratnikov
$Date: 2005/10/20 05:18:37 $
$Revision: 1.2 $
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



unsigned long HcalElectronicsMap::chId2eId (unsigned long fChId, bool fWarning) const {
  const Item* item = findByChId (fChId, fWarning);
  return item ? item->mElId : 0;
}

unsigned long HcalElectronicsMap::chId2tId (unsigned long fChId, bool fWarning) const {
  const Item* item = findByChId (fChId, fWarning);
  return item ? item->mTrigId : 0;
}

unsigned long HcalElectronicsMap::eId2tId (unsigned long fElectronicsId, bool fWarning) const {
  const Item* item = findByElId (fElectronicsId, fWarning);
  return item ? item->mTrigId : 0;
}
unsigned long HcalElectronicsMap::eId2chId (unsigned long fElectronicsId, bool fWarning) const {
  const Item* item = findByElId (fElectronicsId, fWarning);
  return item ? item->mChId : 0;
}
unsigned long HcalElectronicsMap::tId2eId (unsigned long fTriggerId, bool fWarning) const {
  const Item* item = findByTrigId (fTriggerId, fWarning);
  return item ? item->mElId : 0;
}
unsigned long HcalElectronicsMap::tId2chId (unsigned long fTriggerId, bool fWarning) const {
  const Item* item = findByTrigId (fTriggerId, fWarning);
  return item ? item->mChId : 0;
}

bool HcalElectronicsMap::setMapping (unsigned long fChId, unsigned long fElectronicsId, unsigned long fTriggerId) {
  Item item (fChId, fElectronicsId, fTriggerId);
  mItems.push_back (item);
  mSortedByChId = mSortedByElId = mSortedByTrigId = false;
  return true;
}

std::vector <unsigned long> HcalElectronicsMap::allElectronicsId () const {
  std::vector <unsigned long> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) 
    if (item->mElId) allIds.insert (item->mElId);
  return std::vector <unsigned long> (allIds.begin (), allIds.end ());
}

std::vector <unsigned long> HcalElectronicsMap::allDetectorId () const {
  std::vector <unsigned long> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mChId) allIds.insert (item->mChId);
  return std::vector <unsigned long> (allIds.begin (), allIds.end ());
}

std::vector <unsigned long> HcalElectronicsMap::allTriggerId () const {
  std::vector <unsigned long> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mTrigId) allIds.insert (item->mTrigId);
  return std::vector <unsigned long> (allIds.begin (), allIds.end ());
}

bool HcalElectronicsMap::mapEId2tId (unsigned long fElectronicsId, unsigned long fTriggerId) {
  const Item* item = findByElId (fElectronicsId, false);
  if (item) { // record exists
    if (item->mTrigId == 0) {
      ((Item*)item)->mTrigId = fTriggerId; // just cast avoiding long machinery
    } 
    else if (item->mTrigId != fTriggerId) {
      std::cerr << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId << " already mapped to trigger channel " 
		<< item->mTrigId << ". New value " << fTriggerId << " is ignored" << std::endl;
      return false;
    }
    return true;
  }
  else {
    return setMapping (0, fElectronicsId, fTriggerId);
  }
}

bool HcalElectronicsMap::mapEId2chId (unsigned long fElectronicsId, unsigned long fChId) {
  const Item* item = findByElId (fElectronicsId, false);
  if (item) { // record exists
    if (item->mChId == 0) {
      ((Item*)item)->mChId = fChId; // just cast avoiding long machinery
    } 
    else if (item->mChId != fChId) {
      std::cerr << "HcalElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId << " already mapped to channel " 
		<< item->mChId << ". New value " << fChId << " is ignored" << std::endl;
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
