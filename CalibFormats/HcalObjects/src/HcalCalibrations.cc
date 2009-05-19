#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2009/05/06 22:22:44 $
   $Revision: 1.5 $
*/

HcalCalibrations::HcalCalibrations (const float fGain [4], const float fPedestal [4], 
				    const float fRespCorr, const float fTimeCorr, 
				    const float fLUTCorr ) {
  int iCap = 4;
  while (--iCap >= 0) {
    mRespCorrGain [iCap] = fGain [iCap] * fRespCorr;
    mPedestal [iCap] = fPedestal [iCap];
  }
  mRespCorr = fRespCorr;
  mTimeCorr = fTimeCorr;
  mLUTCorr = fLUTCorr;
}
