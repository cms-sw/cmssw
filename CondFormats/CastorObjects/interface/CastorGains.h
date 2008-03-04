#ifndef CastorGains_h
#define CastorGains_h

/** 
\class CastorGains
\author Panos Katsas (UoA)
POOL container to store Gain values 4xCapId
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorGain.h"
#include "DataFormats/DetId/interface/DetId.h"

// 
class CastorGains {
 public:
CastorGains();
~CastorGains();
  /// get array of values for 4 capIds
  const CastorGain* getValues (DetId fId) const;
  /// get value for given capId = 0..3
  float getValue (DetId fId, int fCapId) const;
  /// get list of all available channels
  std::vector<DetId> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// fill values
  bool addValue (DetId fId, const float fValues [4]);
  /// fill values
  bool addValue (DetId fId, float fValue0, float fValue1, float fValue2, float fValue3);
  /// sort values by channelId  
  void sort ();
  // helper typedefs
  typedef CastorGain Item;
  typedef std::vector <Item> Container;
 private:
  Container mItems;
  bool mSorted;
};

#endif
