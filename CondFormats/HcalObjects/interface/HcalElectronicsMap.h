#ifndef HcalElectronicsMap_h
#define HcalElectronicsMap_h

/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store map between detector ID, electronics ID and trigger ID
$Author: ratnikov
$Date: 2007/12/10 18:36:43 $
$Revision: 1.16 $
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
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
// 
class HcalElectronicsMap {
 public:
  HcalElectronicsMap();
  ~HcalElectronicsMap();

  // swap function
  void swap(HcalElectronicsMap& other);
  // copy-ctor
  HcalElectronicsMap(const HcalElectronicsMap& src);
  // copy assignment operator
  HcalElectronicsMap& operator=(const HcalElectronicsMap& rhs);
  // move constructor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  HcalElectronicsMap(HcalElectronicsMap&& other);
#endif

  /// lookup the logical detid associated with the given electronics id
  //return Null item if no such mapping
  const DetId lookup(HcalElectronicsId fId) const;

  /// brief lookup the electronics detid associated with the given logical id
  //return Null item if no such mapping
  const HcalElectronicsId lookup(DetId fId) const;

  /// brief lookup the trigger logical detid associated with the given electronics id
  //return Null item if no such mapping
  const DetId lookupTrigger(HcalElectronicsId fId) const;

  /// brief lookup the electronics detid associated with the given trigger logical id
  //return Null item if no such mapping
  const HcalElectronicsId lookupTrigger(DetId fId) const;

  /// brief lookup the DetId and full electronics id associated with this partial (dcc/spigot/fiber/fiberchan) id
  bool lookup(const HcalElectronicsId pId, HcalElectronicsId& eid, HcalGenericDetId& did) const;
  /// brief lookup the DetId and full electronics id associated with this partial (dcc/spigot/slb/slbchan) id
  bool lookup(const HcalElectronicsId pId, HcalElectronicsId& eid, HcalTrigTowerDetId& did) const;

  std::vector <HcalElectronicsId> allElectronicsId () const;
  std::vector <HcalElectronicsId> allElectronicsIdPrecision() const;
  std::vector <HcalElectronicsId> allElectronicsIdTrigger() const;
  std::vector <HcalGenericDetId> allPrecisionId () const;
  std::vector <HcalTrigTowerDetId> allTriggerId () const;

  // map channels
  bool mapEId2tId (HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2chId (HcalElectronicsId fElectronicsId, DetId fId);
  // sorting
  void sortById () const;
  void sortByTriggerId () const;
  void sort() {}

  class PrecisionItem { 
  public:
    PrecisionItem () {mId = mElId = 0;}
    PrecisionItem (uint32_t fId, uint32_t fElId) 
      : mId (fId), mElId (fElId) {}
    uint32_t mId;
    uint32_t mElId;
  
  COND_SERIALIZABLE;
};
  class TriggerItem { 
  public:
    TriggerItem () {mElId = mTrigId = 0;}
    TriggerItem (uint32_t fTrigId, uint32_t fElId) 
      : mTrigId (fTrigId), mElId (fElId) { }
    uint32_t mTrigId;
    uint32_t mElId;
  
  COND_SERIALIZABLE;
};
 protected:
  const PrecisionItem* findById (unsigned long fId) const;
  const PrecisionItem* findPByElId (unsigned long fElId) const;
  const TriggerItem* findTByElId (unsigned long fElId) const;
  const TriggerItem* findByTrigId (unsigned long fTrigId) const;
  
  std::vector<PrecisionItem> mPItems;
  std::vector<TriggerItem> mTItems;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  mutable std::atomic<std::vector<const PrecisionItem*>*> mPItemsById COND_TRANSIENT;
  mutable std::atomic<std::vector<const TriggerItem*>*> mTItemsByTrigId COND_TRANSIENT;
#else
  mutable std::vector<const PrecisionItem*>* mPItemsById COND_TRANSIENT;
  mutable std::vector<const TriggerItem*>* mTItemsByTrigId COND_TRANSIENT;
#endif

 COND_SERIALIZABLE;
};

#endif
