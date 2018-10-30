#ifndef CASTOR_CALIBRATIONS_H
#define CASTOR_CALIBRATIONS_H

/** \class CastorCalibrations
    
    Container for retrieved calibration constants for Castor

*/
class CastorCalibrations {
 public:
  CastorCalibrations () : mGain{}, mPedestal{} {};
  CastorCalibrations (const float fGain [4], const float fPedestal [4]);
  /// get gain for capid=0..3
  double gain (int fCapId) const {return mGain [fCapId];}
  /// get pedestal for capid=0..3
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
 private:
  double mGain [4];
  double mPedestal [4];
};

#endif
