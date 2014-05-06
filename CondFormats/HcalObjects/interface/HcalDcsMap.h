#ifndef HcalDcsMap_h
#define HcalDcsMap_h

/** 
\class HcalDcsMap
\author Gena Kukartsev
POOL object to store map between detector ID and DCS ID
Inspired by HcalElectronicsMap
$Author: kukartse
$Date: 2007/12/14 13:31:21 $
$Revision: 1.1 $
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
// 
class HcalDcsMap {
 public:
  HcalDcsMap();
  ~HcalDcsMap();
  
  // swap function
  void swap(HcalDcsMap& other);
  // copy-ctor
  HcalDcsMap(const HcalDcsMap& src);
  // copy assignment operator
  HcalDcsMap& operator=(const HcalDcsMap& rhs);
  // move constructor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  HcalDcsMap(HcalDcsMap&& other);
#endif

  // lookup the logical detid associated with the given DCS id
  // return Null item if no such mapping.
  //
  // Note that type (HV, BV etc.) is a part of HcalDcsDetId but
  // it is not preserved in the map in order to reduce data volume.
  // This is possible because the same HCAL channels will correspond
  // to the DCS channels where only the type is different.
  //
  // For the aforementioned reason, you might use any DCS type
  // when constructing the DetId for this lookup
  const std::vector<HcalDetId> lookup(HcalDcsDetId fId) const;
  
  // brief lookup the DCS detid associated with the given logical id
  //return Null item if no such mapping
  //
  // Note that type (HV, BV etc.) is a part of HcalDcsDetId but
  // it is not preserved in the map in order to reduce data volume.
  // This is possible because the same HCAL channels will correspond
  // to the DCS channels where only the type is different.
  //
  // For this reason, you need to specify  the DCS type in order
  // to extract proper HcalDcsDetId from the map
  const std::vector<HcalDcsDetId> lookup(HcalDetId fId, HcalDcsDetId::DcsType type) const;

  // map channels
  // DCS type is a part of DcsDetId but it does not make sense to keep
  // duplicate records in the map for DCS channels where only type is different.
  // Hence, the type in HcalDcsDetId is always forced to DCSUNKNOWN
  // inside this method
  bool mapGeomId2DcsId (HcalDetId fId, HcalDcsDetId fDcsId);

  // sorting
  void sortById () const;
  void sortByDcsId () const;
  void sort() {}

  class Item { 
  public:
    Item () {mId = mDcsId = 0;}
    Item (uint32_t fId, uint32_t fDcsId) 
      : mId (fId), mDcsId (fDcsId) {}
    uint32_t mId;
    uint32_t mDcsId;
  
  COND_SERIALIZABLE;
};

  class const_iterator{
  public:
    friend class HcalDcsMap;
    const_iterator(){}
    ~const_iterator(){}
    bool operator!=(const const_iterator & other);
    const_iterator operator++();
    const_iterator operator++(int);
    void next(void);
    HcalDcsDetId getHcalDcsDetId(void);
    HcalDetId getHcalDetId(void);
  private:
    std::vector<const Item *>::const_iterator fIter;
  };

  // iterators
  const_iterator beginById(void) const;
  const_iterator beginByDcsId(void) const;
  const_iterator endById(void) const;
  const_iterator endByDcsId(void) const;

 protected:
  const std::vector<const Item *> findById (unsigned long fId) const;
  const std::vector<const Item *> findByDcsId (unsigned long fDcsId) const;
  
  // these are inspired by the emap. Not clear if they are needed
  // for this DCS map at all since it's many-to-many map
  std::vector <HcalDcsDetId> allHcalDcsDetId () const;
  std::vector <HcalGenericDetId> allHcalDetId () const;

  std::vector<Item> mItems;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  mutable std::atomic<std::vector<const Item*>*> mItemsById COND_TRANSIENT;
  mutable std::atomic<std::vector<const Item*>*> mItemsByDcsId COND_TRANSIENT;
  const std::vector<const Item*>* getItemsById(void){return mItemsById.load(std::memory_order_acquire);}
  const std::vector<const Item*>* getItemsByDcsId(void){return mItemsByDcsId.load(std::memory_order_acquire);}
#else
  mutable std::vector<const Item*> mItemsById COND_TRANSIENT;
  mutable std::vector<const Item*> mItemsByDcsId COND_TRANSIENT;
  const std::vector<const Item*>* getItemsById(void){return &mItemsById;}
  const std::vector<const Item*>* getItemsByDcsId(void){return &mItemsByDcsId;}
#endif

 COND_SERIALIZABLE;
};

#endif
