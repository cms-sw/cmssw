/** 
\class CastorPedestals
\author Panos Katsas (UoA)
POOL object to store Pedestal values 4xCapId

*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  class compareItems {
  public:
    bool operator () (const CastorPedestals::Item& first, const CastorPedestals::Item& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  CastorPedestals::Container::const_iterator 
  find (const CastorPedestals::Container& container, unsigned long id) {
    CastorPedestals::Container::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

CastorPedestals::CastorPedestals() 
  : mSorted (false) {}

CastorPedestals::~CastorPedestals(){}

const CastorPedestal* CastorPedestals::getValues (DetId fId) const {
  Item target (fId.rawId (), 0, 0, 0, 0);
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "CastorPedestals::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, fId.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != target.rawId ()) 
    throw cms::Exception ("Conditions not found") << "Unavailable Pedestals for cell " << HcalGenericDetId(fId);
  return &(*cell);
}

float CastorPedestals::getValue (DetId fId, int fCapId) const {
  const CastorPedestal* values;
  if (fCapId >= 0 && fCapId < 4) {
    values = getValues (fId);
    if (values) return values->getValue (fCapId);
  }
  else {
    std::cerr << "CastorPedestals::getValue-> capId " << fCapId << " is out of range [0..3]" << std::endl;
  }
  return -1;
}

bool CastorPedestals::addValue (DetId fId, const float fValues [4]) {
  return addValue (fId, fValues [0], fValues [1], fValues [2], fValues [3]);
}

bool CastorPedestals::addValue (DetId fId, float fValue0, float fValue1, float fValue2, float fValue3) {
  Item item (fId.rawId (), fValue0, fValue1, fValue2, fValue3);
  mItems.push_back (item);
  mSorted = false;
  return true;
}

std::vector<DetId> CastorPedestals::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}


void CastorPedestals::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
