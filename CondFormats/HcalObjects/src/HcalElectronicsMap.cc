/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Hcal channels
$Author: ratnikov
$Date: 2006/08/23 20:24:51 $
$Revision: 1.11 $
*/

#include <iostream>
#include <set>

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

HcalElectronicsMap::HcalElectronicsMap() 
  : mSortedById (false),
    mSortedByElId (false),
    mSortedByTrigId (false)
{}

HcalElectronicsMap::~HcalElectronicsMap(){}

const HcalElectronicsMap::Item* HcalElectronicsMap::findById (unsigned long fId, bool fWarning) const {
  Item target (fId, 0, 0);
  std::vector<HcalElectronicsMap::Item>::const_iterator item;
  if (mSortedById) {
    item = std::lower_bound (mItems.begin(), mItems.end(), target, Item::LessById ());
  }
  else {
    if (fWarning) std::cerr << "HcalElectronicsMap-> container is not sorted. Use sortByChaId () to search effectively" << std::endl;
    item = mItems.begin();
    Item::LessById less;
    while (! (item == mItems.end() || (!less (*item,target) && !less (target, *item)))) item++;
    //    while (item != mItems.end() && !( less (*item,target) || less (target, *item))) item++;
  }
  if (item == mItems.end() || item->mId != fId) 
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
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
  if (item == mItems.end() || item->mElId != fElId)
    // throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for e-cell " << fElId;
    return 0;
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
  if (item == mItems.end() || item->mTrigId != fTrigId)
    // throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for trig-cell " << fTrigId;
    return 0;
  return &*item;
}


const DetId HcalElectronicsMap::lookup(HcalElectronicsId fId, bool fWarning ) const {
  const Item* item = findByElId (fId.rawId (), fWarning);
  return DetId (item ? item->mId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookup(DetId fId, bool fWarning) const {
  const Item* item = findById (fId.rawId (), fWarning);
  return HcalElectronicsId (item ? item->mElId : 0);
}

const DetId HcalElectronicsMap::lookupTrigger(HcalElectronicsId fId, bool fWarning) const {
  const Item* item = findByElId (fId.rawId (), fWarning);
  return DetId (item ? item->mTrigId : 0);
}

const HcalElectronicsId HcalElectronicsMap::lookupTrigger(DetId fId, bool fWarning) const {
  const Item* item = findByTrigId (fId.rawId (), fWarning);
  return HcalElectronicsId (item ? item->mElId : 0);
}

bool HcalElectronicsMap::setMapping (DetId fId, HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  Item item (fId.rawId (), fElectronicsId.rawId (), fTriggerId.rawId ());
  mItems.push_back (item);
  mSortedById = mSortedByElId = mSortedByTrigId = false;
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
    return setMapping (DetId (), fElectronicsId, fTriggerId);
  }
}

bool HcalElectronicsMap::mapEId2chId (HcalElectronicsId fElectronicsId, DetId fId) {
  const Item* item = findByElId (fElectronicsId.rawId (), false);
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
  return setMapping (fId, fElectronicsId, 0);
}

void HcalElectronicsMap::sortById () {
    if (!mSortedById) {
    std::sort (mItems.begin(), mItems.end(), Item::LessById ());
    mSortedById = true;
    mSortedByElId = mSortedByTrigId = false;
  }

}
void HcalElectronicsMap::sortByElectronicsId () {
    if (!mSortedByElId) {
    std::sort (mItems.begin(), mItems.end(), Item::LessByElId ());
    mSortedByElId = true;
    mSortedById = mSortedByTrigId = false;
  }

}
void HcalElectronicsMap::sortByTriggerId () {
    if (!mSortedByTrigId) {
    std::sort (mItems.begin(), mItems.end(), Item::LessByTrigId ());
    mSortedByTrigId = true;
    mSortedById = mSortedByElId = false;
  }

}
