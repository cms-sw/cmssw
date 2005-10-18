/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Hcal channels
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.5 $
*/

#include <iostream>

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
    while (item != mItems.end() && !( less (*item,target) || less (target, *item))) item++;
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
