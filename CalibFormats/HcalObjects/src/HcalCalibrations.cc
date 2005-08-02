#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2005/08/01 21:47:49 $
   $Revision: 1.1 $
*/

HcalCalibrations::HcalCalibrations (double fGain [4], double fPedestal [4]) {
  int iCap = 4;
  while (--iCap >= 0) {
    mGain [iCap] = fGain [iCap];
    mPedestal [iCap] = fPedestal [iCap];
  }
}
