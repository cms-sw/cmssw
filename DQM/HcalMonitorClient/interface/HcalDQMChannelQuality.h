#ifndef HcalDQMChannelQuality_h
#define HcalDQMChannelQuality_h

#include <string>
#include <vector>
#include <algorithm>
#include <boost/cstdint.hpp>

// 
class HcalDQMChannelQuality {
 public:
  enum Quality {
    UNKNOWN = 0,
    BAD = 1,
    GOOD = 2,
    COOL = 3,
    HOT = 4,
    DEAD = 5,
    END = 6
  };

  HcalDQMChannelQuality();
  ~HcalDQMChannelQuality();

  Quality quality (unsigned long fId, bool fWarning = true) const;
  bool isMasked (unsigned long fId, bool fWarning = true) const;
  unsigned short flagAlgo (unsigned long fId, bool fWarning = true) const;
  std::string comment (unsigned long fId, bool fWarning = true) const;
  bool maskCell (Quality fQual) const;

  bool sorted () const {return mSorted;}
  // get list of all available channels
  std::vector<unsigned long> getAllChannels () const;
  bool setChannel (unsigned long fId, unsigned short algo, Quality fQuality = UNKNOWN);

  void sort ();
  static const char* str (Quality fQuality);
  class Item {
  public:
    bool operator< (const Item& other) const {return mId < other.mId;}
    bool operator== (const Item& other) const {return mId == other.mId;}
    uint32_t mId;
    short    mAlgo;
    bool mMasked;
    std::string   mComment;
    unsigned mQuality;
  };
 protected:
  std::vector<Item> mItems;
  bool mSorted;
};

#endif

