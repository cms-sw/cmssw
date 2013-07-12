#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include <cstdlib>

/** \class CastorCalibrations
    
    Container for retrieved calibration constants for Castor
   $Author: katsas
   $Date: 2008/03/04 10:01:27 $
   $Revision: 1.1 $
*/

CastorCalibrations::CastorCalibrations (const float fGain [4], const float fPedestal [4]) {
  for (size_t iCap = 0; iCap < 4; ++iCap)
  {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
