/** 
\class HcalPedestalWidths
\author Fedor Ratnikov (UMd)
POOL object to store PedestalWidth values 4xCapId
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.5 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

namespace {
  class compareItems {
  public:
    bool operator () (const HcalPedestalWidths::Item& first, const HcalPedestalWidths::Item& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  HcalPedestalWidths::Container::const_iterator 
  find (const HcalPedestalWidths::Container& container, unsigned long id) {
    HcalPedestalWidths::Container::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

HcalPedestalWidths::HcalPedestalWidths() 
  : mSorted (false) {}

HcalPedestalWidths::~HcalPedestalWidths(){}

const HcalPedestalWidth* HcalPedestalWidths::getValues (HcalDetId fId) const {
  Item target (fId.rawId (), 0, 0, 0, 0);
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "HcalPedestalWidths::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, fId.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != target.rawId ()) return 0;
  return &(*cell);
}

float HcalPedestalWidths::getValue (HcalDetId fId, int fCapId) const {
  const HcalPedestalWidth* values;
  if (fCapId > 0 && fCapId <= 4) {
    values = getValues (fId);
    if (values) return values->getValue (fCapId);
  }
  else {
    std::cerr << "HcalPedestalWidths::getValue-> capId " << fCapId << " is out of range [1..4]" << std::endl;
  }
  return -1;
}

bool HcalPedestalWidths::addValue (HcalDetId fId, const float fValues [4]) {
  return addValue (fId, fValues [0], fValues [1], fValues [2], fValues [3]);
}

bool HcalPedestalWidths::addValue (HcalDetId fId, float fValue1, float fValue2, float fValue3, float fValue4) {
  Item item (fId.rawId (), fValue1, fValue2, fValue3, fValue4);
  mItems.push_back (item);
  mSorted = false;
  return true;
}

std::vector<HcalDetId> HcalPedestalWidths::getAllChannels () const {
  std::vector<HcalDetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (HcalDetId (item->rawId ()));
  }
  return result;
}


void HcalPedestalWidths::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
