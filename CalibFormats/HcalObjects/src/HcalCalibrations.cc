#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include <cstdlib>

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2011/10/13 09:44:45 $
   $Revision: 1.7 $
*/

HcalCalibrations::HcalCalibrations (const float fGain [4], const float fPedestal [4], 
				    const float fRespCorr, const float fTimeCorr, 
				    const float fLUTCorr ) {
  for (size_t iCap = 0; iCap < 4; ++iCap) {
    mRespCorrGain [iCap] = fGain [iCap] * fRespCorr;
    mPedestal [iCap] = fPedestal [iCap];
  }
  mRespCorr = fRespCorr;
  mTimeCorr = fTimeCorr;
  mLUTCorr = fLUTCorr;
}
