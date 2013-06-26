/** 
\class CastorElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store mapping for Castor channels
$Author: ratnikov
$Date: 2009/03/26 18:03:15 $
$Revision: 1.2 $
Adapted for CASTOR by L. Mundim
*/

#include <iostream>
#include <set>

#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CastorElectronicsMap::CastorElectronicsMap() : 
  mPItems(CastorElectronicsId::maxLinearIndex+1),
  mTItems(CastorElectronicsId::maxLinearIndex+1),
  sortedByPId(false),
  sortedByTId(false)
{}

namespace castor_impl {
  class LessById {public: bool operator () (const CastorElectronicsMap::PrecisionItem* a, const CastorElectronicsMap::PrecisionItem* b) {return a->mId < b->mId;}};
  class LessByTrigId {public: bool operator () (const CastorElectronicsMap::TriggerItem* a, const CastorElectronicsMap::TriggerItem* b) {return a->mTrigId < b->mTrigId;}};
}

CastorElectronicsMap::~CastorElectronicsMap(){}

const CastorElectronicsMap::PrecisionItem* CastorElectronicsMap::findById (unsigned long fId) const {
  PrecisionItem target (fId, 0);
  std::vector<const CastorElectronicsMap::PrecisionItem*>::const_iterator item;

  if (!sortedByPId) sortById();
  
  item = std::lower_bound (mPItemsById.begin(), mPItemsById.end(), &target, castor_impl::LessById());
  if (item == mPItemsById.end() || (*item)->mId != fId) 
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const CastorElectronicsMap::PrecisionItem* CastorElectronicsMap::findPByElId (unsigned long fElId) const {
  CastorElectronicsId eid(fElId);
  const PrecisionItem* i=&(mPItems[eid.linearIndex()]);
  
  if (i!=0 && i->mElId!=fElId) i=0;
  return i;
}

const CastorElectronicsMap::TriggerItem* CastorElectronicsMap::findTByElId (unsigned long fElId) const {
  CastorElectronicsId eid(fElId);
  const TriggerItem* i=&(mTItems[eid.linearIndex()]);
  
  if (i!=0 && i->mElId!=fElId) i=0;
  return i;
}


const CastorElectronicsMap::TriggerItem* CastorElectronicsMap::findByTrigId (unsigned long fTrigId) const {
  TriggerItem target (fTrigId,0);
  std::vector<const CastorElectronicsMap::TriggerItem*>::const_iterator item;

  if (!sortedByTId) sortByTriggerId();
  
  item = std::lower_bound (mTItemsByTrigId.begin(), mTItemsByTrigId.end(), &target, castor_impl::LessByTrigId());
  if (item == mTItemsByTrigId.end() || (*item)->mTrigId != fTrigId) 
    //    throw cms::Exception ("Conditions not found") << "Unavailable Electronics map for cell " << fId;
    return 0;
  return *item;
}

const DetId CastorElectronicsMap::lookup(CastorElectronicsId fId ) const {
  const PrecisionItem* item = findPByElId (fId.rawId ());
  return DetId (item ? item->mId : 0);
}

const CastorElectronicsId CastorElectronicsMap::lookup(DetId fId) const {
  const PrecisionItem* item = findById (fId.rawId ());
  return CastorElectronicsId (item ? item->mElId : 0);
}

const DetId CastorElectronicsMap::lookupTrigger(CastorElectronicsId fId) const {
  const TriggerItem* item = findTByElId (fId.rawId ());
  return DetId (item ? item->mTrigId : 0);
}

const CastorElectronicsId CastorElectronicsMap::lookupTrigger(DetId fId) const {
  const TriggerItem* item = findByTrigId (fId.rawId ());
  return CastorElectronicsId (item ? item->mElId : 0);
}

bool CastorElectronicsMap::lookup(const CastorElectronicsId pid, CastorElectronicsId& eid, HcalGenericDetId& did) const {
  const PrecisionItem* i=&(mPItems[pid.linearIndex()]);
  if (i!=0 && i->mId!=0) {
    eid=CastorElectronicsId(i->mElId);
    did=HcalGenericDetId(i->mId);
    return true;
  } else return false;
}

bool CastorElectronicsMap::lookup(const CastorElectronicsId pid, CastorElectronicsId& eid, HcalTrigTowerDetId& did) const {
  const TriggerItem* i=&(mTItems[pid.linearIndex()]);
  if (i!=0 && i->mTrigId!=0) {
    eid=CastorElectronicsId(i->mElId);
    did=HcalGenericDetId(i->mTrigId);
    return true;
  } else return false;  
}


std::vector <CastorElectronicsId> CastorElectronicsMap::allElectronicsId () const {
  std::vector <CastorElectronicsId> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); item != mPItems.end (); item++) 
    if (item->mElId) result.push_back(CastorElectronicsId(item->mElId));
  for (std::vector<TriggerItem>::const_iterator item = mTItems.begin (); item != mTItems.end (); item++) 
    if (item->mElId) result.push_back(CastorElectronicsId(item->mElId));

  return result;
}

std::vector <CastorElectronicsId> CastorElectronicsMap::allElectronicsIdPrecision() const {
  std::vector <CastorElectronicsId> result;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); item != mPItems.end (); item++) 
    if (item->mElId) result.push_back(CastorElectronicsId(item->mElId));
  return result;
}

std::vector <CastorElectronicsId> CastorElectronicsMap::allElectronicsIdTrigger() const {
  std::vector <CastorElectronicsId> result;
  for (std::vector<TriggerItem>::const_iterator item = mTItems.begin (); item != mTItems.end (); item++) 
    if (item->mElId) result.push_back(CastorElectronicsId(item->mElId));

  return result;
}

std::vector <HcalGenericDetId> CastorElectronicsMap::allPrecisionId () const {
  std::vector <HcalGenericDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<PrecisionItem>::const_iterator item = mPItems.begin (); item != mPItems.end (); item++)  
    if (item->mId) allIds.insert (item->mId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++) {
      result.push_back (HcalGenericDetId (*channel));
  }
  return result;
}

std::vector <HcalTrigTowerDetId> CastorElectronicsMap::allTriggerId () const {
  std::vector <HcalTrigTowerDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<TriggerItem>::const_iterator item = mTItems.begin (); item != mTItems.end (); item++)  
    if (item->mTrigId) allIds.insert (item->mTrigId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++)
    result.push_back (HcalTrigTowerDetId (*channel));
  return result;
}

bool CastorElectronicsMap::mapEId2tId (CastorElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId) {
  TriggerItem& item = mTItems[fElectronicsId.linearIndex()];
  sortedByTId=false;
  if (item.mElId==0) item.mElId=fElectronicsId.rawId();
  if (item.mTrigId == 0) {
    item.mTrigId = fTriggerId.rawId (); // just cast avoiding long machinery
  } 
  else if (item.mTrigId != fTriggerId.rawId ()) {
    edm::LogWarning("CASTOR") << "CastorElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId  << " already mapped to trigger channel " 
	      << (HcalTrigTowerDetId(item.mTrigId)) << ". New value " << fTriggerId << " is ignored" ;
    return false;
  }
  return true;
}

bool CastorElectronicsMap::mapEId2chId (CastorElectronicsId fElectronicsId, DetId fId) {
  PrecisionItem& item = mPItems[fElectronicsId.linearIndex()];

  sortedByPId=false;
  if (item.mElId==0) item.mElId=fElectronicsId.rawId();
  if (item.mId == 0) {
    item.mId = fId.rawId ();
  } 
  else if (item.mId != fId.rawId ()) {
     edm::LogWarning("CASTOR") << "CastorElectronicsMap::mapEId2tId-> Electronics channel " <<  fElectronicsId << " already mapped to channel " 
			     << HcalGenericDetId(item.mId) << ". New value " << HcalGenericDetId(fId) << " is ignored" ;
       return false;
  }
  return true;
}

void CastorElectronicsMap::sortById () const {
  if (!sortedByPId) {
    mPItemsById.clear();
    for (std::vector<PrecisionItem>::const_iterator i=mPItems.begin(); i!=mPItems.end(); ++i) {
      if (i->mElId) mPItemsById.push_back(&(*i));
    }
    
    std::sort (mPItemsById.begin(), mPItemsById.end(), castor_impl::LessById ());
    sortedByPId=true;
  }
}

void CastorElectronicsMap::sortByTriggerId () const {
  if (!sortedByTId) {
    mTItemsByTrigId.clear();
    for (std::vector<TriggerItem>::const_iterator i=mTItems.begin(); i!=mTItems.end(); ++i) {
      if (i->mElId) mTItemsByTrigId.push_back(&(*i));
    }
    
    std::sort (mTItemsByTrigId.begin(), mTItemsByTrigId.end(), castor_impl::LessByTrigId ());
    sortedByTId=true;
  }
}
