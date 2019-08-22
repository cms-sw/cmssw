#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include <cstdlib>

/** \class HcalCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for HCAL
   $Author: ratnikov
*/

HcalCalibrationWidths::HcalCalibrationWidths(const float fGain[4],
                                             const float fPedestal[4],
                                             const float fEffectivePedestal[4]) {
  for (size_t iCap = 0; iCap < 4; ++iCap) {
    mGain[iCap] = fGain[iCap];
    mPedestal[iCap] = fPedestal[iCap];
    mEffectivePedestal[iCap] = fEffectivePedestal[iCap];
  }
}
