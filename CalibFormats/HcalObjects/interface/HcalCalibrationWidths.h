#ifndef HCAL_CALIBRATION_WIDTHS_H
#define HCAL_CALIBRATION_WIDTHs_H

/** \class HcalCalibrationWidths
    
    Abstract interface for retrieving uncertainties of calibration 
    constants for HCAL
   $Author: ratnikov
   $Date: 2005/07/27 19:44:31 $
   $Revision: 1.1 $
*/
class HcalCalibrationWidths {
 public:
  virtual double gain (int fCapId) const = 0;
  virtual double pedestal (int fCapId) const = 0;
};

#endif
