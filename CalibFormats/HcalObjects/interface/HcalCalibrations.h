#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2005/10/04 18:05:54 $
   $Revision: 1.3 $
*/
class HcalCalibrations {
 public:
  HcalCalibrations () {};
  HcalCalibrations (const float fGain [4], const float fPedestal [4]);
  double gain (int fCapId) const {return mGain [fCapId];}
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
