#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include <cstdlib>

/** \class CastorCalibrations
    
    Container for retrieved calibration constants for Castor
   $Author: katsas
   $Date: 2011/10/13 09:37:52 $
   $Revision: 1.2 $
*/

CastorCalibrations::CastorCalibrations (const float fGain [4], const float fPedestal [4]) {
  for (size_t iCap = 0; iCap < 4; ++iCap)
  {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
