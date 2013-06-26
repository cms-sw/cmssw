#ifndef HcalRawGains_h
#define HcalRawGains_h

/** 
\class HcalRawGains
\author Fedor Ratnikov (UMd)
POOL container to store Gain values 4xCapId
$Author: ratnikov
$Date: 2006/11/21 03:35:33 $
$Revision: 1.1 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalRawGain.h"
#include "DataFormats/DetId/interface/DetId.h"

// 
class HcalRawGains {
 public:
  HcalRawGains();
  ~HcalRawGains();
  /// get value 
  const HcalRawGain* getValues (DetId fId) const;
  /// get list of all available channels
  std::vector<DetId> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// add new (empty) item
  HcalRawGain* addItem (DetId fId);
  /// fill values
  void addValues (DetId fId, const HcalRawGain& fValues);
  /// sort values by channelId  
  void sort ();
  // helper typedefs
  typedef HcalRawGain Item;
  typedef std::vector <Item> Container;
 private:
  Container mItems;
  bool mSorted;
};

#endif
