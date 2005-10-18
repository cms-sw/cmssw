/** 
\class HcalCannelQuality
\author Fedor Ratnikov (UMd)
POOL object to store pedestal values 4xCapId
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.4 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalCannelQuality.h"

HcalCannelQuality::HcalCannelQuality() 
  : mSorted (false) {}

HcalCannelQuality::~HcalCannelQuality(){}

HcalCannelQuality::Quality HcalCannelQuality::quality (unsigned long fId, bool fWarning) const {
  Item target;
  target.mId = fId;
  std::vector<Item>::const_iterator cell;
  if (sorted ()) {
    cell = std::lower_bound (mItems.begin(), mItems.end(), target);
  }
  else {
    if (fWarning) std::cerr << "HcalCannelQuality::quality-> container is not sorted. Sort it to search effectively" << std::endl;
    cell = std::find (mItems.begin(), mItems.end(), target);
  }
  if (cell == mItems.end() || cell->mId != fId) return UNKNOWN;
  return (HcalCannelQuality::Quality) cell->mQuality;
}

bool HcalCannelQuality::setChannel (unsigned long fId, Quality fQuality) {
  Item item;
  item.mId = fId;
  item.mQuality = unsigned (fQuality);
  mSorted = false;
  return true;
}

void HcalCannelQuality::sort () {
  if (!mSorted) {
    std::sort (mItems.begin(), mItems.end());
    mSorted = true;
  }
}
