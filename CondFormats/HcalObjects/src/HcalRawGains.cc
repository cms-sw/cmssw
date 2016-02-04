/** 
\class HcalRawGains
\author Fedor Ratnikov (UMd)
POOL object to store Gain values 4xCapId
$Author: ratnikov
$Date: 2007/12/04 19:06:24 $
$Revision: 1.4 $
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalRawGains.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  class compareItems {
  public:
    bool operator () (const HcalRawGains::Item& first, const HcalRawGains::Item& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  HcalRawGains::Container::const_iterator 
  find (const HcalRawGains::Container& container, unsigned long id) {
    HcalRawGains::Container::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

HcalRawGains::HcalRawGains() 
  : mSorted (false) {}

HcalRawGains::~HcalRawGains(){}

const HcalRawGain* HcalRawGains::getValues (DetId fId) const {
  Item target (fId.rawId (), 0, 0, 0, HcalRawGain::BAD);
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "HcalRawGains::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, fId.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != target.rawId ())
    throw cms::Exception ("Conditions not found") << "Unavailable Raw Gains for cell " << HcalGenericDetId(target.rawId());
  return &(*cell);
}

std::vector<DetId> HcalRawGains::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}


HcalRawGain* HcalRawGains::addItem (DetId fId) {
  HcalRawGain item (fId.rawId ());
  mItems.push_back (item);
  mSorted = false;
  return &(mItems.back ());
}

void HcalRawGains::addValues (DetId fId, const HcalRawGain& fValues) {
  Item item (fId.rawId (), fValues.getValue(), fValues.getError(), fValues.getVoltage(), fValues.getStatus());
  mItems.push_back (item);
  mSorted = false;
}


void HcalRawGains::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
