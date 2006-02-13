#ifndef HcalElectronicsMap_h
#define HcalElectronicsMap_h

/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store map between detector ID, electronics ID and trigger ID
$Author: ratnikov
$Date: 2006/02/01 21:34:52 $
$Revision: 1.5 $
*/

#include <vector>
#include <algorithm>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

// 
class HcalElectronicsMap {
 public:
  HcalElectronicsMap();
  ~HcalElectronicsMap();

  /// lookup the logical detid associated with the given electronics id
  //return Null item if no such mapping
  const HcalDetId lookup(HcalElectronicsId fId, bool fWarning = true) const;

  /// brief lookup the electronics detid associated with the given logical id
  //return Null item if no such mapping
  const HcalElectronicsId lookup(HcalDetId fId, bool fWarning = true) const;

  /// brief lookup the trigger logical detid associated with the given electronics id
  //return Null item if no such mapping
  const HcalTrigTowerDetId lookupTrigger(HcalElectronicsId fId, bool fWarning = true) const;

  /// brief lookup the electronics detid associated with the given trigger logical id
  //return Null item if no such mapping
  const HcalElectronicsId lookupTrigger(HcalTrigTowerDetId fId, bool fWarning = true) const;

  /// Check if mapping for given Id is defined (also as NA)
  bool known (HcalElectronicsId fId, bool fWarning = true) const;
  bool known (HcalDetId fId, bool fWarning = true) const;
  bool known (HcalTrigTowerDetId fId, bool fWarning = true) const;
  


  std::vector <HcalElectronicsId> allElectronicsId () const;
  std::vector <HcalDetId> allDetectorId () const;
  std::vector <HcalTrigTowerDetId> allTriggerId () const;

  // map channels
  bool setMapping (HcalDetId fChId, HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2tId (HcalElectronicsId fElectronicsId, HcalTrigTowerDetId fTriggerId);
  bool mapEId2chId (HcalElectronicsId fElectronicsId, HcalDetId fChId);
  // sorting
  void sortByChaId ();
  void sortByElectronicsId ();
  void sortByTriggerId ();
  void sort () {sortByElectronicsId ();}
  class Item { 
  public:
    Item () {mChId = mElId = mTrigId = 0;}
    Item (unsigned long fChId, unsigned long fElId, unsigned long fTrigId) 
      : mChId (fChId), mElId (fElId), mTrigId (fTrigId) {}
    class LessByChId {public: bool operator () (const Item& a, const Item& b) {return a.mChId < b.mChId;}};
    class LessByElId {public: bool operator () (const Item& a, const Item& b) {return a.mElId < b.mElId;}};
    class LessByTrigId {public: bool operator () (const Item& a, const Item& b) {return a.mTrigId < b.mTrigId;}};
    unsigned long mChId;
    unsigned long mElId;
    unsigned long mTrigId;
  };
 protected:
  const Item* findByChId (unsigned long fChId, bool fWarning) const;
  const Item* findByElId (unsigned long fElId, bool fWarning) const;
  const Item* findByTrigId (unsigned long fTrigId, bool fWarning) const;
  
  std::vector<Item> mItems;
  bool mSortedByChId;
  bool mSortedByElId;
  bool mSortedByTrigId;
};

#endif
