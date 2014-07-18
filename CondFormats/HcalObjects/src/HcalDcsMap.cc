/** 
\class HcalDcsMap
\author Gena Kukartsev
POOL object to store map between detector ID and DCS ID
Inspired by HcalElectronicsMap
$Author: kukartse
$Date: 2010/02/22 21:08:07 $
$Revision: 1.1 $
*/

#include <iostream>
#include <set>

#include "CondFormats/HcalObjects/interface/HcalDcsMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalDcsMap::HcalDcsMap()
  //FIXME mItems(HcalDcsDetId::maxLinearIndex+1),
  //: mItems(0x7FFF),
    : mItemsById(nullptr), mItemsByDcsId(nullptr)
{
}

HcalDcsMap::~HcalDcsMap(){
    delete mItemsById.load();
    delete mItemsByDcsId.load();
}
// copy-ctor
HcalDcsMap::HcalDcsMap(const HcalDcsMap& src)
    : mItems(src.mItems),
      mItemsById(nullptr), mItemsByDcsId(nullptr) {}
// copy assignment operator
HcalDcsMap&
HcalDcsMap::operator=(const HcalDcsMap& rhs) {
    HcalDcsMap temp(rhs);
    temp.swap(*this);
    return *this;
}
// public swap function
void HcalDcsMap::swap(HcalDcsMap& other) {
    std::swap(mItems, other.mItems);
    other.mItemsByDcsId.exchange(
            mItemsByDcsId.exchange(other.mItemsByDcsId.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
    other.mItemsById.exchange(
            mItemsById.exchange(other.mItemsById.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
// move constructor
HcalDcsMap::HcalDcsMap(HcalDcsMap&& other) 
    : HcalDcsMap() {
    other.swap(*this);
}

namespace hcal_impl {
  class LessById {
  public: 
    bool operator () (const HcalDcsMap::Item* a,
		      const HcalDcsMap::Item* b) {
      return a->mId < b->mId;
    }
  };

  class LessByDcsId {
  public: 
    bool operator () (const HcalDcsMap::Item* a,
		      const HcalDcsMap::Item* b) {
      return a->mDcsId < b->mDcsId;
    }
  };
}

HcalDcsMap::const_iterator HcalDcsMap::beginById(void) const{
  const_iterator _iter;
  sortById();
  _iter.fIter = (*mItemsById.load(std::memory_order_acquire)).begin();
  return _iter;
}

HcalDcsMap::const_iterator HcalDcsMap::beginByDcsId(void) const{
  const_iterator _iter;
  sortByDcsId();
  _iter.fIter = (*mItemsByDcsId.load(std::memory_order_acquire)).begin();
  return _iter;
}

HcalDcsMap::const_iterator HcalDcsMap::endById(void) const{
  const_iterator _iter;
  sortById();
  _iter.fIter = (*mItemsById.load(std::memory_order_acquire)).end();
  return _iter;
}

HcalDcsMap::const_iterator HcalDcsMap::endByDcsId(void) const{
  const_iterator _iter;
  sortByDcsId();
  _iter.fIter = (*mItemsByDcsId.load(std::memory_order_acquire)).end();
  return _iter;
}

// iterator methods
bool HcalDcsMap::const_iterator::operator!=(const HcalDcsMap::const_iterator & other){
  if (fIter != other.fIter) return true;
  else return false;
}

HcalDcsMap::const_iterator HcalDcsMap::const_iterator::operator++(){
  ++fIter;
  return *this;
}

HcalDcsMap::const_iterator HcalDcsMap::const_iterator::operator++(int){
  const_iterator i = *this;
  ++fIter;
  return i;
}

void HcalDcsMap::const_iterator::next(void){
    ++fIter;
}

HcalDcsDetId HcalDcsMap::const_iterator::getHcalDcsDetId(void){
  return (*fIter)->mDcsId;
}

HcalDetId HcalDcsMap::const_iterator::getHcalDetId(void){
  return (*fIter)->mId;
}


const std::vector<const HcalDcsMap::Item *> HcalDcsMap::findById (unsigned long fId) const {
  Item target (fId, 0);
  std::vector<const HcalDcsMap::Item*>::const_iterator item;
  std::vector<const HcalDcsMap::Item *> result;

  sortById();
  
  hcal_impl::LessById lessById;
  auto ptr = (*mItemsById.load(std::memory_order_acquire));
  item = std::lower_bound (ptr.begin(), ptr.end(), &target, lessById);
  if (item == ptr.end() || (*item)->mId != fId){
    //    throw cms::Exception ("Conditions not found") << "Unavailable Dcs map for cell " << fId;
    return result;
  }
  else{
    if(item != ptr.end() && !lessById(&target, *item)){
      result.push_back( *item );
      ++item;
    }
  }
  return result;
}

const std::vector<const HcalDcsMap::Item *> HcalDcsMap::findByDcsId (unsigned long fDcsId) const {
  Item target (0, fDcsId);
  std::vector<const HcalDcsMap::Item*>::const_iterator item;
  std::vector<const HcalDcsMap::Item *> result;

  sortByDcsId();

  hcal_impl::LessByDcsId lessByDcsId;  
  auto ptr = (*mItemsByDcsId.load(std::memory_order_acquire));
  item = std::lower_bound (ptr.begin(), ptr.end(), &target, lessByDcsId);
  if (item == ptr.end() || (*item)->mDcsId != fDcsId) {
    //    throw cms::Exception ("Conditions not found") << "Unavailable Dcs map for cell " << fDcsId;
    return result;
  }
  else{
    if(item != ptr.end() && !lessByDcsId(&target, *item)){
      result.push_back( *item );
      ++item;
    }
  }
  return result;
}

const std::vector<HcalDetId> HcalDcsMap::lookup(HcalDcsDetId fId ) const{
  // DCS type is a part of DcsDetId but it does not make sense to keep
  // duplicate records in the map for DCS channels where only type is different.
  // Hence, the type in HcalDcsDetId is always forced to DCSUNKNOWN
  HcalDcsDetId fDcsId_notype(fId.subdet(),
			     fId.ring(), // side is already included
			     fId.slice(),
			     HcalDcsDetId::DCSUNKNOWN,
			     fId.subchannel());
  const std::vector<const Item *> items = findByDcsId (fDcsId_notype.rawId ());
  std::vector<HcalDetId> _ids;
  for (std::vector<const Item *>::const_iterator item = items.begin();
       item != items.end();
       ++item){
    _ids.push_back( DetId(*item ? (*item)->mId : 0) );
  }
  return _ids;
}

const std::vector<HcalDcsDetId> HcalDcsMap::lookup(HcalDetId fId, HcalDcsDetId::DcsType type) const {
  const std::vector<const Item *> items = findById (fId.rawId ());
  std::vector<HcalDcsDetId> _ids;
  for (std::vector<const Item *>::const_iterator item = items.begin();
       item != items.end();
       ++item){
    HcalDcsDetId _id(*item ? (*item)->mId : 0);
    _ids.push_back( HcalDcsDetId(_id.subdet(),
				 _id.zside()*_id.ring(),
				 _id.slice(),
				 type,
				 _id.subchannel()
				 )
		    );
  }
  return _ids;
}

//FIXME: remove duplicates
std::vector <HcalDcsDetId> HcalDcsMap::allHcalDcsDetId () const {
  std::vector <HcalDcsDetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) 
    if (item->mDcsId) result.push_back(HcalDcsDetId(item->mDcsId));
  return result;
}

// FIXME: remove duplicates
std::vector <HcalGenericDetId> HcalDcsMap::allHcalDetId () const {
  std::vector <HcalGenericDetId> result;
  std::set <unsigned long> allIds;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++)  
    if (item->mId) allIds.insert (item->mId);
  for (std::set <unsigned long>::const_iterator channel = allIds.begin (); channel != allIds.end (); channel++) {
      result.push_back (HcalGenericDetId (*channel));
  }
  return result;
}


bool HcalDcsMap::mapGeomId2DcsId (HcalDetId fId, HcalDcsDetId fDcsId) {
  // DCS type is a part of DcsDetId but it does not make sense to keep
  // duplicate records in the map for DCS channels where only type is different.
  // Hence, the type in HcalDcsDetId is always forced to DCSUNKNOWN
  HcalDcsDetId fDcsId_notype(fDcsId.subdet(),
			     fDcsId.ring(), // side is included
			     fDcsId.slice(),
			     HcalDcsDetId::DCSUNKNOWN,
			     fDcsId.subchannel());
  const std::vector<const Item *> items = findByDcsId(fDcsId_notype);
  for (std::vector<const Item *>::const_iterator item = items.begin();
       item != items.end();
       ++item){
    if ((*item)->mId == fId){
      edm::LogWarning("HCAL") << "HcalDcsMap::mapGeomId2DcsId-> Geom channel " <<  fId 
			      << " already mapped to DCS channel " << fDcsId_notype;
      return false; // element already exists
    }
  }
  Item _item(fId, fDcsId_notype);
  mItems.push_back(_item);
  delete mItemsById.load();
  mItemsById = nullptr;
  delete mItemsByDcsId.load();
  mItemsByDcsId = nullptr;
  return true;
}


void HcalDcsMap::sortById () const {
  if (!mItemsById.load(std::memory_order_acquire)) {
      auto ptr = new std::vector<const Item*>;
      for (auto i=mItems.begin(); i!=mItems.end(); ++i) {
          if (i->mDcsId) ptr->push_back(&(*i));
      }

      std::sort (ptr->begin(), ptr->end(), hcal_impl::LessById ());
      //atomically try to swap this to become mItemsById
      std::vector<const Item*>* expect = nullptr;
      bool exchanged = mItemsById.compare_exchange_strong(expect, ptr, std::memory_order_acq_rel);
      if(!exchanged) {
          delete ptr;
      }
  }
}

void HcalDcsMap::sortByDcsId () const {
  if (!mItemsByDcsId.load(std::memory_order_acquire)) {
      auto ptr = new std::vector<const Item*>;
      for (auto i=mItems.begin(); i!=mItems.end(); ++i) {
          if (i->mDcsId) ptr->push_back(&(*i));
      }
    
      std::sort (ptr->begin(), ptr->end(), hcal_impl::LessByDcsId ());
      //atomically try to swap this to become mItemsByDcsId
      std::vector<const Item*>* expect = nullptr;
      bool exchanged = mItemsByDcsId.compare_exchange_strong(expect, ptr, std::memory_order_acq_rel);
      if(!exchanged) {
          delete ptr;
      }
  }
}
