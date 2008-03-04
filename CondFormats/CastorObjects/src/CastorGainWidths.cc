/** 
\class CastorGainWidths
\author Panos Katsas (UoA)
POOL object to store GainWidth values 4xCapId

*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  class compareItems {
  public:
    bool operator () (const CastorGainWidths::Item& first, const CastorGainWidths::Item& second) const {
      return first.rawId () < second.rawId ();
    }
  };

  CastorGainWidths::Container::const_iterator 
  find (const CastorGainWidths::Container& container, unsigned long id) {
    CastorGainWidths::Container::const_iterator result = container.begin ();
    for (; result != container.end (); result++) {
      if (result->rawId () == id) break; // found
    }
    return result;
  }
}

CastorGainWidths::CastorGainWidths() 
  : mSorted (false) {}

CastorGainWidths::~CastorGainWidths(){}

const CastorGainWidth* CastorGainWidths::getValues (DetId fId) const {
  Item target (fId.rawId (), 0, 0, 0, 0);
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target, compareItems ());
  }
  else {
    std::cerr << "CastorGainWidths::getValues-> container is not sorted. Please sort it to search effectively" << std::endl;
    cell = find (mItems, fId.rawId ());
  }
  if (cell == mItems.end() || cell->rawId () != target.rawId ())
    throw cms::Exception ("Conditions not found") << "Unavailable Gain Widths for cell " << HcalGenericDetId(fId);
  return &(*cell);
}

float CastorGainWidths::getValue (DetId fId, int fCapId) const {
  const CastorGainWidth* values;
  if (fCapId >= 0 && fCapId < 4) {
    values = getValues (fId);
    if (values) return values->getValue (fCapId);
  }
  else {
    std::cerr << "CastorGainWidths::getValue-> capId " << fCapId << " is out of range [0..3]" << std::endl;
  }
  return -1;
}

bool CastorGainWidths::addValue (DetId fId, const float fValues [4]) {
  return addValue (fId, fValues [0], fValues [1], fValues [2], fValues [3]);
}

bool CastorGainWidths::addValue (DetId fId, float fValue0, float fValue1, float fValue2, float fValue3) {
  Item item (fId.rawId (), fValue0, fValue1, fValue2, fValue3);
  mItems.push_back (item);
  mSorted = false;
  return true;
}

std::vector<DetId> CastorGainWidths::getAllChannels () const {
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (DetId (item->rawId ()));
  }
  return result;
}


void CastorGainWidths::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end(), compareItems ());
    mSorted = true;
  }
}
