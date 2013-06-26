#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include <cstdlib>

/** \class HcalCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for HCAL
   $Author: ratnikov
   $Date: 2011/10/13 09:44:45 $
   $Revision: 1.3 $
*/

HcalCalibrationWidths::HcalCalibrationWidths (const float fGain [4], const float fPedestal [4]) {
  for (size_t iCap = 0; iCap < 4; ++iCap)
  {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
