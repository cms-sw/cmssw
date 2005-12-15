/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/10/18 23:34:56 $
$Revision: 1.1 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

namespace {
  int index (int fCapId, int Range) {return fCapId*4+Range;}

  class compareItems {
  public:
    bool operator () (const HcalQIECoder& first, const HcalQIECoder& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  std::vector<HcalQIECoder>::const_iterator 
  find (const std::vector<HcalQIECoder>& container, unsigned long id) {
    std::vector<HcalQIECoder>::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

HcalQIEData::HcalQIEData() 
  : mSorted (false) {}

HcalQIEData::~HcalQIEData(){}

const HcalQIECoder* HcalQIEData::getCoder (HcalDetId fId) const {
  HcalQIECoder target (fId.rawId ());
  std::vector<HcalQIECoder>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "HcalQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, target.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != fId.rawId ()) return 0;
  return &(*cell);
}

std::vector<HcalDetId> HcalQIEData::getAllChannels () const {
  std::vector<HcalDetId> result;
  for (std::vector<HcalQIECoder>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (HcalDetId (item->rawId ()));
  }
  return result;
}

bool HcalQIEData::setShape (const float fLowEdges [32]) {
  return mShape.setLowEdges (fLowEdges);
}

bool HcalQIEData::addCoder (HcalDetId fId, const HcalQIECoder& fCoder) {
  HcalQIECoder newCoder (fId.rawId ());
  for (int range = 0; range < 4; range++) { 
    for (int capid = 0; capid < 4; capid++) {
      newCoder.setOffset (capid, range, fCoder.offset (capid, range));
      newCoder.setSlope (capid, range, fCoder.slope (capid, range));
    }
  }
 mItems.push_back (newCoder);
 mSorted = false;
 return true; 
}

void HcalQIEData::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
