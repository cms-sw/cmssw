#ifndef CastorRawGains_h
#define CastorRawGains_h

/** 
\class CastorRawGains
\author Panos Katsas (UoA)
POOL container to store Gain values 4xCapId
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorRawGain.h"
#include "DataFormats/DetId/interface/DetId.h"

// 
class CastorRawGains {
 public:
  CastorRawGains();
  ~CastorRawGains();
  /// get value 
  const CastorRawGain* getValues (DetId fId) const;
  /// get list of all available channels
  std::vector<DetId> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// add new (empty) item
  CastorRawGain* addItem (DetId fId);
  /// fill values
  void addValues (DetId fId, const CastorRawGain& fValues);
  /// sort values by channelId  
  void sort ();
  // helper typedefs
  typedef CastorRawGain Item;
  typedef std::vector <Item> Container;
 private:
  Container mItems;
  bool mSorted;
};

#endif
