#ifndef HcalCannelQuality_h
#define HcalCannelQuality_h

/** 
\class HcalCannelQuality
\author Fedor Ratnikov (UMd)
POOL object to store channel quality information
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.4 $
*/

#include <vector>
#include <algorithm>

// 
class HcalCannelQuality {
 public:
  enum Quality {
    UNKNOWN = 0,
    BAD = 1,
    GOOD = 2,
    HOT = 3,
    DEAD = 4,
    END = 5
  };
  HcalCannelQuality();
  ~HcalCannelQuality();
  Quality quality (unsigned long fId, bool fWarning = true) const;
  bool sorted () const {return mSorted;}
  bool setChannel (unsigned long fId, Quality fQuality = UNKNOWN);
  void sort ();
 protected:
  class Item {
  public:
    bool operator< (const Item& other) const {return mId < other.mId;}
    bool operator== (const Item& other) const {return mId == other.mId;}
    unsigned long mId;
    unsigned mQuality;
  };
  std::vector<Item> mItems;
  bool mSorted;
};

#endif
