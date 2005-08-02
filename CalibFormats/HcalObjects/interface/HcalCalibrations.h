#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2005/08/01 21:47:49 $
   $Revision: 1.1 $
*/
class HcalCalibrations {
 public:
  HcalCalibrations (double fGain [4], double fPedestal [4]);
  double gain (int fCapId) const {return mGain [fCapId];}
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
