/** 
\class CastorCalibrationQIEData
\author Panos Katsas (UoA)
POOL object to store pedestal values 4xCapId
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorCalibrationQIEData.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  class compareItems {
  public:
    bool operator () (const CastorCalibrationQIECoder& first, const CastorCalibrationQIECoder& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  std::vector<CastorCalibrationQIECoder>::const_iterator 
  find (const std::vector<CastorCalibrationQIECoder>& container, unsigned long id) {
    std::vector<CastorCalibrationQIECoder>::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

CastorCalibrationQIEData::CastorCalibrationQIEData() 
  : mSorted (true) {}

CastorCalibrationQIEData::~CastorCalibrationQIEData(){}

const CastorCalibrationQIECoder* CastorCalibrationQIEData::getCoder (DetId fId) const {
  CastorCalibrationQIECoder target (fId.rawId ());
  std::vector<CastorCalibrationQIECoder>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "CastorCalibrationQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, target.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != fId.rawId ()) 
    throw cms::Exception ("Conditions not found") << "Unavailable Coder for cell " << HcalGenericDetId(fId);
  return &(*cell);
}

std::vector<DetId> CastorCalibrationQIEData::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<CastorCalibrationQIECoder>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}

bool CastorCalibrationQIEData::addCoder (DetId fId, const CastorCalibrationQIECoder& fCoder) {
  CastorCalibrationQIECoder newCoder (fId.rawId ());
  newCoder.setMinCharges (fCoder.minCharges ());
  mItems.push_back (newCoder);
  mSorted = false;
  return true; 
}

void CastorCalibrationQIEData::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
    std::cout << "CastorCalibrationQIEData::sort ()->" << mItems.size () << std::endl; 
  }
}
