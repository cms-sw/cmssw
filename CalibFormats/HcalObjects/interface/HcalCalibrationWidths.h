#ifndef HCAL_CALIBRATION_WIDTHS_H
#define HCAL_CALIBRATION_WIDTHS_H

/** \class HcalCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for HCAL
   $Author: ratnikov
*/
class HcalCalibrationWidths {
public:
  HcalCalibrationWidths(){};
  HcalCalibrationWidths(const float fGain[4], const float fPedestal[4], const float fEffectivePedestal[4]);
  /// get gain width for capid=0..3
  double gain(int fCapId) const { return mGain[fCapId]; }
  /// get pedestal width for capid=0..3
  double pedestal(int fCapId) const { return mPedestal[fCapId]; }
  /// get effective pedestal width for capid=0..3
  double effpedestal(int fCapId) const { return mEffectivePedestal[fCapId]; }

private:
  double mGain[4];
  double mPedestal[4];
  double mEffectivePedestal[4];
};
#endif
