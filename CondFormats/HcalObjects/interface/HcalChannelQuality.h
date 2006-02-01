#ifndef HcalChannelQuality_h
#define HcalChannelQuality_h

/** 
\class HcalChannelQuality
\author Fedor Ratnikov (UMd)
POOL object to store channel quality information
$Author: ratnikov
$Date: 2005/11/02 21:54:11 $
$Revision: 1.2 $
*/

#include <vector>
#include <algorithm>

// 
class HcalChannelQuality {
 public:
  enum Quality {
    UNKNOWN = 0,
    BAD = 1,
    GOOD = 2,
    HOT = 3,
    DEAD = 4,
    END = 5
  };
  HcalChannelQuality();
  ~HcalChannelQuality();
  Quality quality (unsigned long fId, bool fWarning = true) const;
  bool sorted () const {return mSorted;}
  // get list of all available channels
  std::vector<unsigned long> getAllChannels () const;
  bool setChannel (unsigned long fId, Quality fQuality = UNKNOWN);
  void sort ();
  static const char* str (Quality fQuality);
  class Item {
  public:
    bool operator< (const Item& other) const {return mId < other.mId;}
    bool operator== (const Item& other) const {return mId == other.mId;}
    unsigned long mId;
    unsigned mQuality;
  };
 protected:
  std::vector<Item> mItems;
  bool mSorted;
};

#endif

