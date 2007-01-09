/** 
\class HcalCalibrationQIEData
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2006/08/10 22:51:50 $
$Revision: 1.3 $
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"

namespace {
  class compareItems {
  public:
    bool operator () (const HcalCalibrationQIECoder& first, const HcalCalibrationQIECoder& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  std::vector<HcalCalibrationQIECoder>::const_iterator 
  find (const std::vector<HcalCalibrationQIECoder>& container, unsigned long id) {
    std::vector<HcalCalibrationQIECoder>::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

HcalCalibrationQIEData::HcalCalibrationQIEData() 
  : mSorted (true) {}

HcalCalibrationQIEData::~HcalCalibrationQIEData(){}

const HcalCalibrationQIECoder* HcalCalibrationQIEData::getCoder (DetId fId) const {
  HcalCalibrationQIECoder target (fId.rawId ());
  std::vector<HcalCalibrationQIECoder>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "HcalCalibrationQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, target.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != fId.rawId ()) 
    throw cms::Exception ("Conditions not found") << "Unavailable Coder for cell " << fId.rawId();
  return &(*cell);
}

std::vector<DetId> HcalCalibrationQIEData::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<HcalCalibrationQIECoder>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}

bool HcalCalibrationQIEData::addCoder (DetId fId, const HcalCalibrationQIECoder& fCoder) {
  HcalCalibrationQIECoder newCoder (fId.rawId ());
  newCoder.setMinCharges (fCoder.minCharges ());
  mItems.push_back (newCoder);
  mSorted = false;
  return true; 
}

void HcalCalibrationQIEData::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
    std::cout << "HcalCalibrationQIEData::sort ()->" << mItems.size () << std::endl; 
  }
}
