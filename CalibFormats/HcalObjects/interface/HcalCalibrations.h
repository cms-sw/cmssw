#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2005/12/15 23:37:59 $
   $Revision: 1.4 $
*/
class HcalCalibrations {
 public:
  HcalCalibrations () {};
  HcalCalibrations (const float fGain [4], const float fPedestal [4]);
  /// get gain for capid=0..3
  double gain (int fCapId) const {return mGain [fCapId];}
  /// get pedestal for capid=0..3
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
