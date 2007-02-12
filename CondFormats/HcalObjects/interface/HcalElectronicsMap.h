#ifndef HcalElectronicsMap_h
#define HcalElectronicsMap_h

/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store map between detector ID, electronics ID and trigger ID
$Author: ratnikov
$Date: 2007/01/09 22:49:20 $
$Revision: 1.11 $
*/

#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

// 
class HcalElectronicsMap {
 public:
  HcalElectronicsMap();
  ~HcalElectronicsMap();

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


  std::vector <HcalElectronicsId> allElectronicsId () const;
  std::vector <DetId> allDetectorId () const;
  std::vector <HcalDetId> allHcalDetectorId () const;
  std::vector <HcalCalibDetId> allCalibrationId () const;
  std::vector <HcalTrigTowerDetId> allTriggerId () const;

  // map channels
  bool setMapping (DetId fId, HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2tId (HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2chId (HcalElectronicsId fElectronicsId, DetId fId);
  // sorting
  void sortById () const;
  void sortByElectronicsId ();
  void sortByTriggerId () const;
  void sort () {sortByElectronicsId ();}
  class Item { 
  public:
    Item () {mId = mElId = mTrigId = 0;}
    Item (uint32_t fId, uint32_t fElId, uint32_t fTrigId) 
      : mId (fId), mElId (fElId), mTrigId (fTrigId) {}
    uint32_t mId;
    uint32_t mElId;
    uint32_t mTrigId;
  };
 protected:
  const Item* findById (unsigned long fId) const;
  const Item* findByElId (unsigned long fElId) const;
  const Item* findByTrigId (unsigned long fTrigId) const;
  
  std::vector<Item> mItems;
  mutable std::vector<const Item*> mItemsById;
  mutable std::vector<const Item*> mItemsByTrigId;
};

#endif
