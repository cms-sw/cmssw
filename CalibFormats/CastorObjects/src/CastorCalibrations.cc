#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include <cstdlib>

/** \class CastorCalibrations
    
    Container for retrieved calibration constants for Castor
   $Author: katsas
*/

CastorCalibrations::CastorCalibrations(const float fGain[4], const float fPedestal[4]) {
  for (size_t iCap = 0; iCap < 4; ++iCap) {
    mGain[iCap] = fGain[iCap];
    mPedestal[iCap] = fPedestal[iCap];
  }
}
