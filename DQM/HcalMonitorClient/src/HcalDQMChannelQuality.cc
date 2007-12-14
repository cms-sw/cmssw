#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "DQM/HcalMonitorClient/interface/HcalDQMChannelQuality.h"

HcalDQMChannelQuality::HcalDQMChannelQuality() 
  : mSorted (false) {}

HcalDQMChannelQuality::~HcalDQMChannelQuality(){}

HcalDQMChannelQuality::Quality HcalDQMChannelQuality::quality (unsigned long fId, bool fWarning) const {
  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    if (fWarning) std::cerr << "HcalDQMChannelQuality::quality-> container is not sorted. Sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId)
    throw cms::Exception ("Conditions not found") << "Unavailable Quality for cell " << fId;
  return (HcalDQMChannelQuality::Quality) cell->mQuality;
}

bool HcalDQMChannelQuality::isMasked (unsigned long fId, bool fWarning) const{

  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    if (fWarning) std::cerr << "HcalDQMChannelQuality::quality-> container is not sorted. Sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId)
    throw cms::Exception ("Conditions not found") << "Unavailable Quality for cell " << fId;
  return cell->mMasked;
  
}

unsigned short HcalDQMChannelQuality::flagAlgo (unsigned long fId, bool fWarning) const{

  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    if (fWarning) std::cerr << "HcalDQMChannelQuality::quality-> container is not sorted. Sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId)
    throw cms::Exception ("Conditions not found") << "Unavailable Quality for cell " << fId;
  return cell->mAlgo;

}

std::vector<unsigned long> HcalDQMChannelQuality::getAllChannels () const {
  std::vector<unsigned long> result;
  for (std::vector<Item>::const_iterator item = mItems.begin (); item != mItems.end (); item++) {
    result.push_back (item->mId);
  }
  return result;
}

bool HcalDQMChannelQuality::setChannel (unsigned long fId, unsigned short fAlgo, Quality fQuality) {
  Item item;
  item.mId = fId;
  item.mAlgo = fAlgo;
  item.mMasked = maskCell(fQuality);
  item.mQuality = unsigned (fQuality);
  mSorted = false;
  return true;
}

bool HcalDQMChannelQuality::maskCell(Quality fQuality) const {
  switch (fQuality) {
  case BAD: return true;
  case HOT: return  true;
  case GOOD: return false;
  case COOL: return false;
  case DEAD: return false;
  case END: return false;
  default: return false;
  }
}

void HcalDQMChannelQuality::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end());
    mSorted = true;
  }
}

const char* HcalDQMChannelQuality::str (Quality fQuality) {
  switch (fQuality) {
  case BAD: return "BAD";
  case GOOD: return "GOOD";
  case COOL: return "COOL";
  case HOT: return "HOT";
  case DEAD: return "DEAD";
  case END: return "END";
  default: return "UNKNOWN";
  }
}
