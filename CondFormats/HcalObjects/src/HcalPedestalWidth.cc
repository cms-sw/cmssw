/** 
\class HcalPedestalWidth
\author Fedor Ratnikov (UMd)
correlation matrix for pedestals
$Author: ratnikov
$Date: 2005/12/15 23:38:04 $
$Revision: 1.5 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

namespace {
  int offset (int fCapId1, int fCapId2) {
    static int offsets [4] = {0, 1, 3, 6};
    if (fCapId1 < fCapId2) { // swap
      int tmp = fCapId1; fCapId1 = fCapId2; fCapId2 = tmp;
    }
    return offsets [fCapId1-1] + (fCapId2-1);
  }
}

HcalPedestalWidth::HcalPedestalWidth (int fId) : mId (fId) {
  for (int i = 10; --i >= 0; *(&mSigma11 + i) = 0) {}
}

float HcalPedestalWidth::getWidth (int fCapId) const {
  return sqrt (*(getValues () + offset (fCapId, fCapId)));
}

float HcalPedestalWidth::getSigma (int fCapId1, int fCapId2) const {
  return *(getValues () + offset (fCapId1, fCapId2));
}

void HcalPedestalWidth::setSigma (int fCapId1, int fCapId2, float fSigma) {
  *(&mSigma11 + offset (fCapId1, fCapId2)) = fSigma;
}

