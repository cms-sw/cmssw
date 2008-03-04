#ifndef CastorPedestals_h
#define CastorPedestals_h

/** 
\class CastorPedestals
\author Panos Katsas (UoA)
POOL container to store Pedestal values 4xCapId
$Author: katsas
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorPedestal.h"
#include "DataFormats/DetId/interface/DetId.h"

// 
class CastorPedestals {
 public:
  CastorPedestals();
  ~CastorPedestals();
  /// get array of values for 4 capIds
  const CastorPedestal* getValues (DetId fId) const;
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
  typedef CastorPedestal Item;
  typedef std::vector <Item> Container;
 private:
  Container mItems;
  bool mSorted;
};

#endif
