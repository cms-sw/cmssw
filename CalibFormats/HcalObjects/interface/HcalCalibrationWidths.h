#ifndef HCAL_CALIBRATION_WIDTHS_H
#define HCAL_CALIBRATION_WIDTHs_H

/** \class HcalCalibrationWidths
    
    Container for retrieving uncertainties of calibration constants for HCAL
   $Author: ratnikov
   $Date: 2005/08/02 01:31:24 $
   $Revision: 1.2 $
*/
class HcalCalibrationWidths {
 public:
 public:
  HcalCalibrationWidths (const float fGain [4], const float fPedestal [4]);
  double gain (int fCapId) const {return mGain [fCapId];}
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
