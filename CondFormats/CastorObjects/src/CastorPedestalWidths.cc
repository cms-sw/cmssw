/** 
\class CastorPedestalWidths
\author Panos Katsas (UoA)
POOL object to store PedestalWidth values 4xCapId
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  class compareItems {
  public:
    bool operator () (const CastorPedestalWidths::Item& first, const CastorPedestalWidths::Item& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  std::vector <CastorPedestalWidth>::const_iterator 
  find (const std::vector <CastorPedestalWidth>& container, unsigned long id) {
    std::vector <CastorPedestalWidth>::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

CastorPedestalWidths::CastorPedestalWidths() 
  : mSorted (false) {}

CastorPedestalWidths::~CastorPedestalWidths(){}

const CastorPedestalWidth* CastorPedestalWidths::getValues (DetId fId) const {
  CastorPedestalWidth target (fId.rawId ());
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "CastorPedestalWidths::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, fId.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != target.rawId ())
    throw cms::Exception ("Conditions not found") << "Unavailable PedestalWidth for cell " << HcalGenericDetId(fId);
  return &(*cell);
}

float CastorPedestalWidths::getWidth (DetId fId, int fCapId) const {
  const CastorPedestalWidth* values;
  if (fCapId >= 0 && fCapId < 4) {
    values = getValues (fId);
    if (values) return values->getWidth (fCapId);
  }
  else {
    std::cerr << "CastorPedestalWidths::getWidth-> capId " << fCapId << " is out of range [0..3]" << std::endl;
  }
  return -1;
}

float CastorPedestalWidths::getSigma (DetId fId, int fCapId1, int fCapId2) const {
  const CastorPedestalWidth* values;
  if (fCapId1 >= 0 && fCapId1 < 4 && fCapId2 >= 0 && fCapId2 < 4) {
    values = getValues (fId);
    if (values) return values->getSigma (fCapId1, fCapId2);
  }
  else {
    std::cerr << "CastorPedestalWidths::getSigma-> capId " << fCapId1 << " or " << fCapId2 << " is out of range [0..3]" << std::endl;
  }
  return -1;
}

CastorPedestalWidth* CastorPedestalWidths::setWidth (DetId fId) {
  CastorPedestalWidth item (fId.rawId ());
  mItems.push_back (item);
  mSorted = false;
  return &(mItems.back ());
}

void CastorPedestalWidths::setWidth (const CastorPedestalWidth& fItem) {
  mItems.push_back (fItem);
  mSorted = false;
}

std::vector<DetId> CastorPedestalWidths::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}


void CastorPedestalWidths::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
