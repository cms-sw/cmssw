#ifndef HcalPedestalWidths_h
#define HcalPedestalWidths_h

/** 
\class HcalPedestalWidths
\author Fedor Ratnikov (UMd)
POOL container to store PedestalWidth values 4xCapId
$Author: ratnikov
$Date: 2006/04/13 22:40:41 $
$Revision: 1.7 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

#include "DataFormats/DetId/interface/DetId.h"

// 
class HcalPedestalWidths {
 public:
  HcalPedestalWidths();
  ~HcalPedestalWidths();

  /// get all values
  const HcalPedestalWidth* getValues (DetId fId) const;
  /// get value for given capId = 0..3
  float getWidth (DetId fId, int fCapId) const;
  /// get correlation for given capId1/2 = 0..3
  float getSigma (DetId fId, int fCapId1, int fCapId2) const;
  /// get list of all available channels
  std::vector<DetId> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// add new (empty) item
  HcalPedestalWidth* setWidth (DetId fId);
  /// add new (empty) item
  void setWidth (const HcalPedestalWidth& fItem);
  /// sort values by channelId  
  void sort ();
  typedef HcalPedestalWidth Item;
 private:
  std::vector <HcalPedestalWidth> mItems;
  bool mSorted;
};

#endif
