#ifndef HCAL_CALIBRATIONS_H
#define HCAL_CALIBRATIONS_H

/** \class HcalCalibrations
    
    Abstract interface for retrieving calibration 
    constants for HCAL
   $Author: ratnikov
   $Date: 2005/07/27 19:44:31 $
   $Revision: 1.1 $
*/
class HcalCalibrations {
 public:
  virtual double gain (int fCapId) const = 0;
  virtual double pedestal (int fCapId) const = 0;
};

#endif
