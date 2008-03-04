#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"

/** \class CastorCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for Castor
   $Author: katsas
   $Date: 2008/02/15 15:31:58 $
   $Revision: 1.1 $
*/

CastorCalibrationWidths::CastorCalibrationWidths (const float fGain [4], const float fPedestal [4]) {
  int iCap = 4;
  while (--iCap >= 0) {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
