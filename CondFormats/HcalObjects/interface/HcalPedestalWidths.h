#ifndef HcalPedestalWidths_h
#define HcalPedestalWidths_h

/** 
\class HcalPedestalWidths
\author Fedor Ratnikov (UMd)
POOL container to store PedestalWidth values 4xCapId
$Author: ratnikov
$Date: 2005/12/15 23:38:04 $
$Revision: 1.5 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

#ifndef __REFLEX__ 
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#endif
// 
class HcalPedestalWidths {
 public:
  HcalPedestalWidths();
  ~HcalPedestalWidths();

#ifndef __REFLEX__ 
  /// get all values
  const HcalPedestalWidth* getValues (HcalDetId fId) const;
  /// get value for given capId = 1..4
  float getWidth (HcalDetId fId, int fCapId) const;
  /// get correlation for given capId1/2 = 1..4
  float getSigma (HcalDetId fId, int fCapId1, int fCapId2) const;
  /// get list of all available channels
  std::vector<HcalDetId> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// add new (empty) item
  HcalPedestalWidth* setWidth (HcalDetId fId);
  /// add new (empty) item
  void setWidth (const HcalPedestalWidth& fItem);
  /// sort values by channelId  
  void sort ();
#endif
  typedef HcalPedestalWidth Item;
 private:
  std::vector <HcalPedestalWidth> mItems;
  bool mSorted;
};

#endif
