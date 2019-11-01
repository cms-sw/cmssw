#ifndef CastorElectronicsMap_h
#define CastorElectronicsMap_h

/** 
\class CastorElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store map between detector ID, electronics ID and trigger ID
$Author: ratnikov 
$Date: 2007/12/14 13:31:21 $
$Revision: 1.17 $
Modified for CASTOR by L. Mundim
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/CastorElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <cstdint>
//
class CastorElectronicsMap {
public:
  CastorElectronicsMap();
  ~CastorElectronicsMap();

  // swap function
  void swap(CastorElectronicsMap& other);
  // copy-ctor
  CastorElectronicsMap(const CastorElectronicsMap& src);  // copy assignment operator
  CastorElectronicsMap& operator=(const CastorElectronicsMap& rhs);
  // move constructor
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  CastorElectronicsMap(CastorElectronicsMap&& other);
#endif

  /// lookup the logical detid associated with the given electronics id
  //return Null item if no such mapping
  const DetId lookup(CastorElectronicsId fId) const;

  /// brief lookup the electronics detid associated with the given logical id
  //return Null item if no such mapping
  const CastorElectronicsId lookup(DetId fId) const;

  /// brief lookup the trigger logical detid associated with the given electronics id
  //return Null item if no such mapping
  const DetId lookupTrigger(CastorElectronicsId fId) const;

  /// brief lookup the electronics detid associated with the given trigger logical id
  //return Null item if no such mapping
  const CastorElectronicsId lookupTrigger(DetId fId) const;

  /// brief lookup the DetId and full electronics id associated with this partial (dcc/spigot/fiber/fiberchan) id
  bool lookup(const CastorElectronicsId pId, CastorElectronicsId& eid, HcalGenericDetId& did) const;
  /// brief lookup the DetId and full electronics id associated with this partial (dcc/spigot/slb/slbchan) id
  bool lookup(const CastorElectronicsId pId, CastorElectronicsId& eid, HcalTrigTowerDetId& did) const;

  std::vector<CastorElectronicsId> allElectronicsId() const;
  std::vector<CastorElectronicsId> allElectronicsIdPrecision() const;
  std::vector<CastorElectronicsId> allElectronicsIdTrigger() const;
  std::vector<HcalGenericDetId> allPrecisionId() const;
  std::vector<HcalTrigTowerDetId> allTriggerId() const;

  // map channels
  bool mapEId2tId(CastorElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2chId(CastorElectronicsId fElectronicsId, DetId fId);
  // sorting
  void sortById() const;
  void sortByTriggerId() const;
  void sort() {}

  class PrecisionItem {
  public:
    PrecisionItem() { mId = mElId = 0; }
    PrecisionItem(uint32_t fId, uint32_t fElId) : mId(fId), mElId(fElId) {}
    uint32_t mId;
    uint32_t mElId;

    COND_SERIALIZABLE;
  };
  class TriggerItem {
  public:
    TriggerItem() { mElId = mTrigId = 0; }
    TriggerItem(uint32_t fTrigId, uint32_t fElId) : mTrigId(fTrigId), mElId(fElId) {}
    uint32_t mTrigId;
    uint32_t mElId;

    COND_SERIALIZABLE;
  };

protected:
  const PrecisionItem* findById(unsigned long fId) const;
  const PrecisionItem* findPByElId(unsigned long fElId) const;
  const TriggerItem* findTByElId(unsigned long fElId) const;
  const TriggerItem* findByTrigId(unsigned long fTrigId) const;

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
