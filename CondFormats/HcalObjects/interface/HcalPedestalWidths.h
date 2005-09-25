#ifndef HcalPedestalWidths_h
#define HcalPedestalWidths_h

/** 
\class HcalPedestalWidths
\author Fedor Ratnikov (UMd)
POOL object to store pedestal widths values 4xCapId
$Author: ratnikov
$Date: 2005/08/02 01:31:24 $
$Revision: 1.2 $
*/

#include <vector>
#include <algorithm>

// 
class HcalPedestalWidths {
 public:
  HcalPedestalWidths();
  ~HcalPedestalWidths();
  /// get array of values for 4 capIds
  const float* getValues (int fId) const;
  /// get value for given capId = 1..4
  float getValue (int fId, int fCapId) const;
  /// get list of all available channels
  std::vector<int> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// fill values
  bool addValue (int fId, const float fValues [4]);
  /// fill values
  bool addValue (int fId, float fValue1, float fValue2, float fValue3, float fValue4);
  /// sort values by channelId  
  void sort ();
 protected:
  class Item {
  public:
    bool operator< (const Item& other) const {return mId < other.mId;}
    bool operator== (const Item& other) const {return mId == other.mId;}
    int mId;
    float mValue1;
    float mValue2;
    float mValue3;
    float mValue4;
  };
  std::vector<Item> mItems;
  bool mSorted;
};

#endif
