#ifndef HcalChannelStatuses_h
#define HcalChannelStatuses_h


/*
\class HcalChannelStatuses
\author Radek Ofierzynski
POOL object to store HcalChannelStatus
*/

#include <vector>
#include <algorithm>

#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"

class HcalChannelStatuses
{
 public:
  HcalChannelStatuses();
  ~HcalChannelStatuses();

  // get status for a particular cell:
  const HcalChannelStatus* getItem(DetId id) const;

  // set status for a particular cell:
  bool addItem(HcalChannelStatus& mystatus);

  // validity check: is threshold available for a particular cell ?
  bool isEmpty(DetId id);

  // are the statuses sorted wrt. the channels ?
  bool sorted() const {return mSorted;}
  // sort object:
  void sort();
  // get a list of available channels:
  std::vector<DetId> getAllChannels() const;

  // helper typedefs:
  typedef HcalChannelStatus Item;
  typedef std::vector<Item> Container;

 private:
  Container mItems;
  bool mSorted;
};









#endif
