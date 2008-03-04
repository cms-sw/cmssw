#ifndef CastorChannelQuality_h
#define CastorChannelQuality_h

/** 
\class CastorChannelQuality
\author Panos Katsas (UoA)
POOL object to store channel quality information
*/

#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>

// 
class CastorChannelQuality {
 public:
  enum Quality {
    UNKNOWN = 0,
    BAD = 1,
    GOOD = 2,
    HOT = 3,
    DEAD = 4,
    END = 5
  };
  CastorChannelQuality();
  ~CastorChannelQuality();
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
    uint32_t mId;
    unsigned mQuality;
  };
 protected:
  std::vector<Item> mItems;
  bool mSorted;
};

#endif

