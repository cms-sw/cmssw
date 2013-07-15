#ifndef HCAL_CALIBRATION_WIDTHS_H
#define HCAL_CALIBRATION_WIDTHS_H

/** \class HcalCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for HCAL
   $Author: ratnikov
   $Date: 2006/04/13 22:40:40 $
   $Revision: 1.5 $
*/
class HcalCalibrationWidths {
 public:
  HcalCalibrationWidths () {};
  HcalCalibrationWidths (const float fGain [4], const float fPedestal [4]);
  /// get gain width for capid=0..3
  double gain (int fCapId) const {return mGain [fCapId];}
  /// get pedestal width for capid=0..3
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
