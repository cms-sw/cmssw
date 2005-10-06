#ifndef HcalPedestals_h
#define HcalPedestals_h

/** 
\class HcalPedestal
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/09/27 19:01:19 $
$Revision: 1.3 $
*/

#include <vector>
#include <algorithm>

// 
class HcalPedestals {
 public:
  HcalPedestals();
  ~HcalPedestals();
  /// get array of values for 4 capIds
  const float* getValues (unsigned long fId) const;
  /// get value for given capId = 1..4
  float getValue (unsigned long fId, int fCapId) const;
  /// get list of all available channels
  std::vector<unsigned long> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// fill values
  bool addValue (unsigned long fId, const float fValues [4]);
  /// fill values
  bool addValue (unsigned long fId, float fValue1, float fValue2, float fValue3, float fValue4);
  /// sort values by channelId  
  void sort ();
 protected:
  class Item {
  public:
    bool operator< (const Item& other) const {return mId < other.mId;}
    bool operator== (const Item& other) const {return mId == other.mId;}
    unsigned long mId;
    float mValue1;
    float mValue2;
    float mValue3;
    float mValue4;
  };
  std::vector<Item> mItems;
  bool mSorted;
};

#endif
