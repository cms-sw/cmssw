/*
\class HcalChannelStatuses
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds

*/

#include <iostream>
#include "CondFormats/HcalObjects/interface/HcalChannelStatuses.h"


namespace
{
  class compareItems
  {
  public:
    bool operator () (const HcalChannelStatuses::Item& first, const HcalChannelStatuses::Item& second) const
    {
      return first.rawId() < second.rawId();
    }
  };

  HcalChannelStatuses::Container::const_iterator
  find (const HcalChannelStatuses::Container& mycontainer, unsigned long myid)
  {
    HcalChannelStatuses::Container::const_iterator result = mycontainer.begin();
    for (; result != mycontainer.end(); result++)
      {
	if (result->rawId() == myid) break;
      }
    return result;
  }
}


HcalChannelStatuses::HcalChannelStatuses()
  : mSorted(false) {}

HcalChannelStatuses::~HcalChannelStatuses() {}


const HcalChannelStatus* HcalChannelStatuses::getItem(DetId id) const
{
  Item target (id.rawId(), 0);
  std::vector<Item>::const_iterator cell;

  if (sorted())
    {
      cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems());
    }
  else
    {
      std::cerr << "HcalChannelStatuses::getValue: container is not sorted. Please sort it to search effectively" << std::endl;
      cell = find (mItems, id.rawId());
    }

  if (cell == mItems.end() || cell->rawId() != target.rawId()) // not found
    return (new HcalChannelStatus());
  else 
    {
      const HcalChannelStatus* mystatus = &(*cell);
      return mystatus;
    }
}

bool HcalChannelStatuses::addItem(HcalChannelStatus& mystatus)
{
  mItems.push_back(mystatus);
  mSorted = false;
  return true;
}

bool HcalChannelStatuses::isEmpty(DetId id)
{
  const HcalChannelStatus* myItem = getItem(id);
  return myItem->isEmpty();  
}

void HcalChannelStatuses::sort()
{
  if (!mSorted)
    {
      std::sort (mItems.begin(), mItems.end(), compareItems());
      mSorted = true;
    }
}

std::vector<DetId> HcalChannelStatuses::getAllChannels() const
{
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator cell = mItems.begin(); cell != mItems.end(); cell++)
    result.push_back( DetId(cell->rawId()) );

  return result;
}
