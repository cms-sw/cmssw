/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2006/07/29 00:21:33 $
$Revision: 1.4 $
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

namespace {
  HcalQIEShape shape_; // use one default set

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
  : mSorted (true) {}

HcalQIEData::HcalQIEData(const HcalQIEData& a) {
  //std::cout << "HcalQIEData::HcalQIEData-> from:" << a.mItems.size () << std::endl;
  mItems = a.mItems;
  mSorted = a.mSorted;
  //std::cout << "HcalQIEData::HcalQIEData-> to:" << mItems.size () << std::endl;
}

HcalQIEData::~HcalQIEData(){}

const HcalQIEShape& HcalQIEData::getShape () const {
  return shape_;
}

const HcalQIECoder* HcalQIEData::getCoder (DetId fId) const {
  HcalQIECoder target (fId.rawId ());
  std::vector<HcalQIECoder>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "HcalQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, target.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != fId.rawId ())
    throw cms::Exception ("Conditions not found") << "Unavailable QIE data for cell " << fId.rawId();
  return &(*cell);
}

std::vector<DetId> HcalQIEData::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<HcalQIECoder>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}

bool HcalQIEData::addCoder (DetId fId, const HcalQIECoder& fCoder) {
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
    std::cout << "HcalQIEData::sort ()->" << mItems.size () << std::endl; 
  }
}
