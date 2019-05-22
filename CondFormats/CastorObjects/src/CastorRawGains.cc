/** 
\class CastorRawGains
\author Panos Katsas (UoA)
POOL object to store Gain values 4xCapId
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorRawGains.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

namespace {
  class compareItems {
  public:
    bool operator()(const CastorRawGains::Item& first, const CastorRawGains::Item& second) const {
      return first.rawId() < second.rawId();
    }
  };

  CastorRawGains::Container::const_iterator find(const CastorRawGains::Container& container, unsigned long id) {
    CastorRawGains::Container::const_iterator result = container.begin();
    for (; result != container.end(); result++) {
      if (result->rawId() == id)
        break;  // found
    }
    return result;
  }
}  // namespace

CastorRawGains::CastorRawGains() : mSorted(false) {}

CastorRawGains::~CastorRawGains() {}

const CastorRawGain* CastorRawGains::getValues(DetId fId) const {
  Item target(fId.rawId(), 0, 0, 0, CastorRawGain::BAD);
  std::vector<Item>::const_iterator cell;
  if (sorted()) {
    cell = std::lower_bound(mItems.begin(), mItems.end(), target, compareItems());
  } else {
    std::cerr << "CastorRawGains::getValues-> container is not sorted. Please sort it to search effectively"
              << std::endl;
    cell = find(mItems, fId.rawId());
  }
  if (cell == mItems.end() || cell->rawId() != target.rawId())
    throw cms::Exception("Conditions not found")
        << "Unavailable Raw Gains for cell " << HcalGenericDetId(target.rawId());
  return &(*cell);
}

std::vector<DetId> CastorRawGains::getAllChannels() const {
  std::vector<DetId> result;
  for (std::vector<Item>::const_iterator item = mItems.begin(); item != mItems.end(); item++) {
    result.push_back(DetId(item->rawId()));
  }
  return result;
}

CastorRawGain* CastorRawGains::addItem(DetId fId) {
  CastorRawGain item(fId.rawId());
  mItems.push_back(item);
  mSorted = false;
  return &(mItems.back());
}

void CastorRawGains::addValues(DetId fId, const CastorRawGain& fValues) {
  Item item(fId.rawId(), fValues.getValue(), fValues.getError(), fValues.getVoltage(), fValues.getStatus());
  mItems.push_back(item);
  mSorted = false;
}

void CastorRawGains::sort() {
  if (!mSorted) {
    std::sort(mItems.begin(), mItems.end(), compareItems());
    mSorted = true;
  }
}
