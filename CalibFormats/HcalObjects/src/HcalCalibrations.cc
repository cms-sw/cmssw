#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2008/03/03 21:41:48 $
   $Revision: 1.3 $
*/

HcalCalibrations::HcalCalibrations (const float fGain [4], const float fPedestal [4], 
				    const float fRespCorr) {
  int iCap = 4;
  while (--iCap >= 0) {
    mRespCorrGain [iCap] = fGain [iCap] * fRespCorr;
    mPedestal [iCap] = fPedestal [iCap];
  }
  mRespCorr = fRespCorr;
}
