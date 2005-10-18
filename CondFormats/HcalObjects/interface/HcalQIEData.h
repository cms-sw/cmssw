#ifndef HcalQIEData_h
#define HcalQIEData_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE parameters
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.5 $
*/

#include <vector>
#include <algorithm>

// 
class HcalQIEData {
 public:
  HcalQIEData();
  ~HcalQIEData();
  /// get array of values for 4 range x 4 capIds
  const float* getOffsets (unsigned long fId) const;
  const float* getSlopes (unsigned long fId) const;
  /// get list of all available channels
  std::vector<unsigned long> getAllChannels () const;
  /// check if data are sorted
  bool sorted () const {return mSorted;}
  /// fill values [capid][range]
  bool addValue (unsigned long fId, const float fOffsets [16], const float fSlopes [16]);
  /// sort values by channelId  
  void sort ();
 protected:
  class Item {
  public:
    bool operator< (const Item& other) const {return mId < other.mId;}
    bool operator== (const Item& other) const {return mId == other.mId;}
    unsigned long mId;
    float mOffset00;
    float mOffset01;
    float mOffset02;
    float mOffset03;
    float mOffset10;
    float mOffset11;
    float mOffset12;
    float mOffset13;
    float mOffset20;
    float mOffset21;
    float mOffset22;
    float mOffset23;
    float mOffset30;
    float mOffset31;
    float mOffset32;
    float mOffset33;
    float mSlope00;
    float mSlope01;
    float mSlope02;
    float mSlope03;
    float mSlope10;
    float mSlope11;
    float mSlope12;
    float mSlope13;
    float mSlope20;
    float mSlope21;
    float mSlope22;
    float mSlope23;
    float mSlope30;
    float mSlope31;
    float mSlope32;
    float mSlope33;
  };
  std::vector<Item> mItems;
  bool mSorted;
};

#endif
