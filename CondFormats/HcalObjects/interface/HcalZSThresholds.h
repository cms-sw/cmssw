#ifndef HcalZSThresholds_h
#define HcalZSThresholds_h

/*
\class HcalZSThresholds
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds

*/

#include <vector>
#include <algorithm>

#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/HcalObjects/interface/HcalZSThreshold.h"

class HcalZSThresholds
{
 public:
  HcalZSThresholds();
  ~HcalZSThresholds();

  // get threshold for a particular cell:
  const HcalZSThreshold* getItem(DetId id) const;
  const int getValue(DetId id) const;

  // set threshold for a particular cell:
  bool addValue(DetId id, int level);

  // validity check: is threshold available for a particular cell ?
  bool isEmpty(DetId id);

  // are the thresholds sorted wrt. the channels ?
  bool sorted() const {return mSorted;}
  // sort object:
  void sort();
  // get a list of available channels:
  std::vector<DetId> getAllChannels() const;

  // helper typedefs:
  typedef HcalZSThreshold Item;
  typedef std::vector<Item> Container;

 private:
  Container mItems;
  bool mSorted;
};

#endif
