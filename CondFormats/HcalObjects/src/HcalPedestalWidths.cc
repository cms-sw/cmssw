/** 
\class HcalPedestalWidths
\author Fedor Ratnikov (UMd)
POOL object to store PedestalWidth values 4xCapId
$Author: ratnikov
$Date: 2006/08/10 22:51:50 $
$Revision: 1.9 $
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

namespace {
  class compareItems {
  public:
    bool operator () (const HcalPedestalWidths::Item& first, const HcalPedestalWidths::Item& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  std::vector <HcalPedestalWidth>::const_iterator 
  find (const std::vector <HcalPedestalWidth>& container, unsigned long id) {
    std::vector <HcalPedestalWidth>::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

HcalPedestalWidths::HcalPedestalWidths() 
  : mSorted (false) {}

HcalPedestalWidths::~HcalPedestalWidths(){}

const HcalPedestalWidth* HcalPedestalWidths::getValues (DetId fId) const {
  HcalPedestalWidth target (fId.rawId ());
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "HcalPedestalWidths::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, fId.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != target.rawId ())
    throw cms::Exception ("Conditions not found") << "Unavailable PedestalWidth for cell " << target.rawId();
  return &(*cell);
}

float HcalPedestalWidths::getWidth (DetId fId, int fCapId) const {
  const HcalPedestalWidth* values;
  if (fCapId >= 0 && fCapId < 4) {
    values = getValues (fId);
    if (values) return values->getWidth (fCapId);
  }
  else {
    std::cerr << "HcalPedestalWidths::getWidth-> capId " << fCapId << " is out of range [0..3]" << std::endl;
  }
  return -1;
}

float HcalPedestalWidths::getSigma (DetId fId, int fCapId1, int fCapId2) const {
  const HcalPedestalWidth* values;
  if (fCapId1 >= 0 && fCapId1 < 4 && fCapId2 >= 0 && fCapId2 < 4) {
    values = getValues (fId);
    if (values) return values->getSigma (fCapId1, fCapId2);
  }
  else {
    std::cerr << "HcalPedestalWidths::getSigma-> capId " << fCapId1 << " or " << fCapId2 << " is out of range [0..3]" << std::endl;
  }
  return -1;
}

HcalPedestalWidth* HcalPedestalWidths::setWidth (DetId fId) {
  HcalPedestalWidth item (fId.rawId ());
  mItems.push_back (item);
  mSorted = false;
  return &(mItems.back ());
}

void HcalPedestalWidths::setWidth (const HcalPedestalWidth& fItem) {
  mItems.push_back (fItem);
  mSorted = false;
}

std::vector<DetId> HcalPedestalWidths::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}


void HcalPedestalWidths::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
