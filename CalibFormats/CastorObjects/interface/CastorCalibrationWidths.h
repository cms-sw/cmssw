#ifndef CASTOR_CALIBRATION_WIDTHS_H
#define CASTOR_CALIBRATION_WIDTHS_H

/** \class CastorCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for Castor

*/
class CastorCalibrationWidths {
 public:
  CastorCalibrationWidths () : mGain{}, mPedestal{} {};
  CastorCalibrationWidths (const float fGain [4], const float fPedestal [4]);
  /// get gain width for capid=0..3
  double gain (int fCapId) const {return mGain [fCapId];}
  /// get pedestal width for capid=0..3
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
