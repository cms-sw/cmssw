/** 
\class CastorChannelQuality
\author Panos Katsas (UoA)
POOL object to store pedestal values 4xCapId
*/

#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

CastorChannelQuality::CastorChannelQuality() 
  : mSorted (false) {}

CastorChannelQuality::~CastorChannelQuality(){}

CastorChannelQuality::Quality CastorChannelQuality::quality (unsigned long fId, bool fWarning) const {
  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    if (fWarning) std::cerr << "CastorChannelQuality::quality-> container is not sorted. Sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId)
    throw cms::Exception ("Conditions not found") << "Unavailable Quality for cell " << HcalGenericDetId(fId);
  return (CastorChannelQuality::Quality) cell->mQuality;
}

std::vector<unsigned long> CastorChannelQuality::getAllChannels () const {
  std::vector<unsigned long> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (item->mId);
  }
  return result;
}

bool CastorChannelQuality::setChannel (unsigned long fId, Quality fQuality) {
  Item item;
  item.mId = fId;
  item.mQuality = unsigned (fQuality);
  mSorted = false;
  return true;
}

void CastorChannelQuality::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end());
    mSorted = true;
  }
}

const char* CastorChannelQuality::str (Quality fQuality) {
  switch (fQuality) {
  case BAD: return "BAD";
  case GOOD: return "GOOD";
  case HOT: return "HOT";
  case DEAD: return "DEAD";
  case END: return "END";
  default: return "UNKNOWN";
  }
}
