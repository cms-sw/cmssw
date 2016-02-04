#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2009/05/19 16:06:05 $
   $Revision: 1.9 $
*/
class HcalCalibrations {
 public:
  HcalCalibrations () {};
  HcalCalibrations (const float fGain [4], const float fPedestal [4], const float fRespCorr, const float fTimeCorr, const float fLUTCorr);
  /// get LUT corrected and response corrected gain for capid=0..3
  double LUTrespcorrgain (int fCapId) const {return (mLUTCorr *  mRespCorrGain [fCapId]);}
  /// get response corrected gain for capid=0..3
  double respcorrgain (int fCapId) const {return mRespCorrGain [fCapId];}
  /// get raw gain for capid=0..3
  double rawgain (int fCapId) const {return mRespCorrGain [fCapId] / mRespCorr;}
  /// get pedestal for capid=0..3
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
  /// get response correction factor
  double respcorr () const {return mRespCorr;}
  /// get time correction factor
  double timecorr () const {return mTimeCorr;}
 private:
  double mRespCorrGain [4];
  double mPedestal [4];
  double mRespCorr;
  double mTimeCorr;
  double mLUTCorr;
};

#endif
