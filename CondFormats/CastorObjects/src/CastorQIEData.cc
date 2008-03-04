/** 
\class CastorQIEData
\author Panos Katsas (UoA)
POOL object to store pedestal values 4xCapId
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  CastorQIEShape shape_; // use one default set

  int index (int fCapId, int Range) {return fCapId*4+Range;}

  class compareItems {
  public:
    bool operator () (const CastorQIECoder& first, const CastorQIECoder& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  std::vector<CastorQIECoder>::const_iterator 
  find (const std::vector<CastorQIECoder>& container, unsigned long id) {
    std::vector<CastorQIECoder>::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

CastorQIEData::CastorQIEData() 
  : mSorted (true) {}

CastorQIEData::CastorQIEData(const CastorQIEData& a) {
  //std::cout << "CastorQIEData::CastorQIEData-> from:" << a.mItems.size () << std::endl;
  mItems = a.mItems;
  mSorted = a.mSorted;
  //std::cout << "CastorQIEData::CastorQIEData-> to:" << mItems.size () << std::endl;
}

CastorQIEData::~CastorQIEData(){}

const CastorQIEShape& CastorQIEData::getShape () const {
  return shape_;
}

const CastorQIECoder* CastorQIEData::getCoder (DetId fId) const {
  CastorQIECoder target (fId.rawId ());
  std::vector<CastorQIECoder>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "CastorQIEData::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, target.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != fId.rawId ())
    throw cms::Exception ("Conditions not found") << "Unavailable QIE data for cell " << HcalGenericDetId(fId);
  return &(*cell);
}

std::vector<DetId> CastorQIEData::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<CastorQIECoder>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}

bool CastorQIEData::addCoder (DetId fId, const CastorQIECoder& fCoder) {
  CastorQIECoder newCoder (fId.rawId ());
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

void CastorQIEData::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
