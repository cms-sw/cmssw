#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"
#include <cstdlib>

/** \class CastorCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for Castor
   $Author: katsas
   $Date: 2008/03/04 10:01:27 $
   $Revision: 1.1 $
*/

CastorCalibrationWidths::CastorCalibrationWidths (const float fGain [4], const float fPedestal [4]) {
  for (size_t iCap = 0; iCap < 4; ++iCap)
  {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
