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

#include <set>
#include <vector>
#include <algorithm>
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDcsDetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <cstdint>

//forward declaration
namespace HcalDcsMapAddons {
  class Helper;
}

class HcalDcsMap {
public:
  class Item {
  public:
    Item() { mId = mDcsId = 0; }
    Item(uint32_t fId, uint32_t fDcsId) : mId(fId), mDcsId(fDcsId) {}
    uint32_t mId;
    uint32_t mDcsId;

    COND_SERIALIZABLE;
  };

  HcalDcsMap() {}
  HcalDcsMap(const HcalDcsMapAddons::Helper& helper);
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
  HcalDetId lookup(HcalDcsDetId fId) const;

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
  HcalDcsDetId lookup(HcalDetId fId, HcalDcsDetId::DcsType type) const;

  class const_iterator {
  public:
    friend class HcalDcsMap;
    const_iterator() {}
    ~const_iterator() {}
    bool operator!=(const const_iterator& other);
    const_iterator operator++();
    const_iterator operator++(int);
    void next(void);
    HcalDcsDetId getHcalDcsDetId(void);
    HcalDetId getHcalDetId(void);

  private:
    std::vector<const Item*>::const_iterator fIter;
  };

  // iterators
  const_iterator beginById(void) const;
  const_iterator beginByDcsId(void) const;
  const_iterator endById(void) const;
  const_iterator endByDcsId(void) const;

  void initialize();

  const Item* findById(unsigned long fId) const;
  const Item* findByDcsId(unsigned long fDcsId) const;

  //sorting
  void sortById();
  void sortByDcsId();

protected:
  // these are inspired by the emap. Not clear if they are needed
  // for this DCS map at all since it's many-to-many map
  std::vector<HcalDcsDetId> allHcalDcsDetId() const;
  std::vector<HcalGenericDetId> allHcalDetId() const;

  std::vector<Item> mItems;
  std::vector<const Item*> mItemsById COND_TRANSIENT;
  std::vector<const Item*> mItemsByDcsId COND_TRANSIENT;

  COND_SERIALIZABLE;
};

namespace HcalDcsMapAddons {
  class LessById {
  public:
    bool operator()(const HcalDcsMap::Item* a, const HcalDcsMap::Item* b) const { return a->mId < b->mId; }
    bool operator()(const HcalDcsMap::Item& a, const HcalDcsMap::Item& b) const { return a.mId < b.mId; }
    bool equal(const HcalDcsMap::Item* a, const HcalDcsMap::Item* b) const { return a->mId == b->mId; }
    bool good(const HcalDcsMap::Item& a) const { return a.mDcsId; }
  };
  class LessByDcsId {
  public:
    bool operator()(const HcalDcsMap::Item* a, const HcalDcsMap::Item* b) const { return a->mDcsId < b->mDcsId; }
    bool operator()(const HcalDcsMap::Item& a, const HcalDcsMap::Item& b) const { return a.mDcsId < b.mDcsId; }
    bool equal(const HcalDcsMap::Item* a, const HcalDcsMap::Item* b) const { return a->mDcsId == b->mDcsId; }
    bool good(const HcalDcsMap::Item& a) const { return a.mDcsId; }
  };
  class Helper {
  public:
    Helper();
    // map channels
    // DCS type is a part of DcsDetId but it does not make sense to keep
    // duplicate records in the map for DCS channels where only type is different.
    // Hence, the type in HcalDcsDetId is always forced to DCSUNKNOWN
    // inside this method
    bool mapGeomId2DcsId(HcalDetId fId, HcalDcsDetId fDcsId);

    std::set<HcalDcsMap::Item, LessByDcsId> mItems;
  };
}  // namespace HcalDcsMapAddons

#endif
