#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
*/
class HcalCalibrations {
public:
  constexpr HcalCalibrations()
      : mRespCorrGain{0, 0, 0, 0},
        mPedestal{0, 0, 0, 0},
        mEffectivePedestal{0, 0, 0, 0},
        mRespCorr(0),
        mTimeCorr(0),
        mLUTCorr(0) {}
  constexpr HcalCalibrations(const float fGain[4],
                             const float fPedestal[4],
                             const float fEffectivePedestal[4],
                             const float fRespCorr,
                             const float fTimeCorr,
                             const float fLUTCorr)
      : mRespCorrGain{0, 0, 0, 0},
        mPedestal{0, 0, 0, 0},
        mEffectivePedestal{0, 0, 0, 0},
        mRespCorr(0),
        mTimeCorr(0),
        mLUTCorr(0) {
    for (auto iCap = 0; iCap < 4; ++iCap) {
      mRespCorrGain[iCap] = fGain[iCap] * fRespCorr;
      mPedestal[iCap] = fPedestal[iCap];
      mEffectivePedestal[iCap] = fEffectivePedestal[iCap];
    }
    mRespCorr = fRespCorr;
    mTimeCorr = fTimeCorr;
    mLUTCorr = fLUTCorr;
  }
  /// get LUT corrected and response corrected gain for capid=0..3
  constexpr double LUTrespcorrgain(int fCapId) const { return (mLUTCorr * mRespCorrGain[fCapId]); }
  /// get response corrected gain for capid=0..3
  constexpr double respcorrgain(int fCapId) const { return mRespCorrGain[fCapId]; }
  /// get raw gain for capid=0..3
  constexpr double rawgain(int fCapId) const { return mRespCorrGain[fCapId] / mRespCorr; }
  /// get pedestal for capid=0..3
  constexpr double pedestal(int fCapId) const { return mPedestal[fCapId]; }
  /// get effective pedestal for capid=0..3
  constexpr double effpedestal(int fCapId) const { return mEffectivePedestal[fCapId]; }
  /// get response correction factor
  constexpr double respcorr() const { return mRespCorr; }
  /// get time correction factor
  constexpr double timecorr() const { return mTimeCorr; }

private:
  double mRespCorrGain[4];
  double mPedestal[4];
  double mEffectivePedestal[4];
  double mRespCorr;
  double mTimeCorr;
  double mLUTCorr;
};

#endif
