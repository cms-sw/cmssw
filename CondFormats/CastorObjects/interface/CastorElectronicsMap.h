#ifndef CastorElectronicsMap_h
#define CastorElectronicsMap_h

/** 
\class CastorElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store map between detector ID, electronics ID and trigger ID
$Author: ratnikov 
$Date: 2009/03/26 18:03:15 $
$Revision: 1.2 $
Modified for CASTOR by L. Mundim
*/

#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/CastorElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
// 
class CastorElectronicsMap {
 public:
  CastorElectronicsMap();
  ~CastorElectronicsMap();

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

  std::vector <CastorElectronicsId> allElectronicsId () const;
  std::vector <CastorElectronicsId> allElectronicsIdPrecision() const;
  std::vector <CastorElectronicsId> allElectronicsIdTrigger() const;
  std::vector <HcalGenericDetId> allPrecisionId () const;
  std::vector <HcalTrigTowerDetId> allTriggerId () const;

  // map channels
  bool mapEId2tId (CastorElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2chId (CastorElectronicsId fElectronicsId, DetId fId);
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
  };
  class TriggerItem { 
  public:
    TriggerItem () {mElId = mTrigId = 0;}
    TriggerItem (uint32_t fTrigId, uint32_t fElId) 
      : mTrigId (fTrigId), mElId (fElId) { }
    uint32_t mTrigId;
    uint32_t mElId;
  };
 protected:
  const PrecisionItem* findById (unsigned long fId) const;
  const PrecisionItem* findPByElId (unsigned long fElId) const;
  const TriggerItem* findTByElId (unsigned long fElId) const;
  const TriggerItem* findByTrigId (unsigned long fTrigId) const;
  
  std::vector<PrecisionItem> mPItems;
  std::vector<TriggerItem> mTItems;
  mutable std::vector<const PrecisionItem*> mPItemsById;
  mutable bool sortedByPId;
  mutable std::vector<const TriggerItem*> mTItemsByTrigId;
  mutable bool sortedByTId;
};

#endif
