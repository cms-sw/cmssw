#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Container for retrieved calibration constants for HCAL
   $Author: ratnikov
   $Date: 2006/04/13 22:40:40 $
   $Revision: 1.5 $
*/
class HcalCalibrations {
 public:
  HcalCalibrations () {};
  HcalCalibrations (const float fGain [4], const float fPedestal [4], const float fRespCorr);
  /// get gain for capid=0..3
  double gain (int fCapId) const {return mGain [fCapId];}
  /// get pedestal for capid=0..3
  double pedestal (int fCapId) const {return mPedestal [fCapId];}
  /// get response correction factor
  double respcorr () const {return mRespCorr;}
 private:
  double mGain [4];
  double mPedestal [4];
  double mRespCorr;
};

#endif
