#ifndef HcalElectronicsMap_h
#define HcalElectronicsMap_h

/** 
\class HcalElectronicsMap
\author Fedor Ratnikov (UMd)
POOL object to store map between detector ID, electronics ID and trigger ID
$Author: ratnikov
$Date: 2005/10/18 23:34:56 $
$Revision: 1.1 $
*/

#include <vector>
#include <algorithm>

// 
class HcalElectronicsMap {
 public:
  HcalElectronicsMap();
  ~HcalElectronicsMap();
  // Electronics ID
  unsigned long chId2eId (unsigned long fChId, bool fWarning = true) const;
  unsigned long tId2eId (unsigned long fTriggerId, bool fWarning = true) const;
  // Trigger ID
  unsigned long chId2tId (unsigned long fChId, bool fWarning = true) const;
  unsigned long eId2tId (unsigned long fElectronicsId, bool fWarning = true) const;
  // Channel ID
  unsigned long eId2chId (unsigned long fElectronicsId, bool fWarning = true) const;
  unsigned long tId2chId (unsigned long fTriggerId, bool fWarning = true) const;

  std::vector <unsigned long> allElectronicsId () const;

  // map channels
  bool setMapping (unsigned long fChId, unsigned long fElectronicsId, unsigned long fTriggerId);
  bool mapEId2tId (unsigned long fElectronicsId, unsigned long fTriggerId);
  bool mapEId2chId (unsigned long fElectronicsId, unsigned long fChId);
  // sorting
  void sortByChaId ();
  void sortByElectronicsId ();
  void sortByTriggerId ();
  void sort () {sortByElectronicsId ();}
 protected:
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
  const Item* findByChId (unsigned long fChId, bool fWarning) const;
  const Item* findByElId (unsigned long fElId, bool fWarning) const;
  const Item* findByTrigId (unsigned long fTrigId, bool fWarning) const;
  
  std::vector<Item> mItems;
  bool mSortedByChId;
  bool mSortedByElId;
  bool mSortedByTrigId;
};

#endif
