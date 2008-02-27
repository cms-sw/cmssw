/*
\class HcalZSThresholds
\author Radek Ofierzynski
POOL object to store Zero Suppression Thresholds

*/

#include <iostream>
#include "CondFormats/HcalObjects/interface/HcalZSThresholds.h"


namespace
{
  class compareItems
  {
  public:
    bool operator () (const HcalZSThresholds::Item& first, const HcalZSThresholds::Item& second) const
    {
      return first.rawId() < second.rawId();
    }
  };

  HcalZSThresholds::Container::const_iterator
  find (const HcalZSThresholds::Container& mycontainer, unsigned long myid)
  {
    HcalZSThresholds::Container::const_iterator result = mycontainer.begin();
    for (; result != mycontainer.end(); result++)
      {
	if (result->rawId() == myid) break;
      }
    return result;
  }
}


HcalZSThresholds::HcalZSThresholds()
  : mSorted(false) {}

HcalZSThresholds::~HcalZSThresholds() {}


const HcalZSThreshold* HcalZSThresholds::getItem(DetId id) const
{
  Item target (id.rawId(), 0);
  std::vector<Item>::const_iterator cell;

  if (sorted())
    {
      cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems());
    }
  else
    {
      std::cerr << "HcalZSThresholds::getValue: container is not sorted. Please sort it to search effectively" << std::endl;
      cell = find (mItems, id.rawId());
    }

  if (cell == mItems.end() || cell->rawId() != target.rawId()) // not found
    return (new HcalZSThreshold());
  else return &(*cell);
}

const int HcalZSThresholds::getValue(DetId id) const
{
  const HcalZSThreshold* myItem = getItem(id);
  return myItem->getValue();
}

bool HcalZSThresholds::addValue(DetId id, int level)
{
  Item myitem(id,level);
  mItems.push_back(myitem);
  mSorted = false;
  return true;
}

bool HcalZSThresholds::isEmpty(DetId id)
{
  const HcalZSThreshold* myItem = getItem(id);
  return myItem->isEmpty();  
}

void HcalZSThresholds::sort()
{
  if (!mSorted)
    {
      std::sort (mItems.begin(), mItems.end(), compareItems());
      mSorted = true;
    }
}

std::vector<DetId> HcalZSThresholds::getAllChannels() const
{
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator cell = mItems.begin(); cell != mItems.end(); cell++)
    result.push_back( DetId(cell->rawId()) );

  return result;
}
