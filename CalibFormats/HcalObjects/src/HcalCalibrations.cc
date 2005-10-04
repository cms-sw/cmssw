#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2005/08/02 01:31:24 $
   $Revision: 1.1 $
*/

HcalCalibrations::HcalCalibrations (const float fGain [4], const float fPedestal [4]) {
  int iCap = 4;
  while (--iCap >= 0) {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
