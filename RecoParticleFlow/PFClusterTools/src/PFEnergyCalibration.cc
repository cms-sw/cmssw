#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include <math.h>

using namespace std;

PFEnergyCalibration::PFEnergyCalibration() {

//--- initialize calibration parameters
//    for energy correction applied to energy deposits of electrons 
//    and photons in ECAL
  paramECAL_slope_ = 1.;
  paramECAL_offset_ = 0.;
  
//--- initialize calibration parameters
//    for energy correction applied to energy deposits of hadrons in HCAL
  paramHCAL_slope_ = 2.17;
  paramHCAL_offset_ = 1.73;
  paramHCAL_damping_ = 2.49;

//--- initialize calibration parameters
//    for energy correction applied to combined energy deposits of hadrons in HCAL and ECAL
  paramECALplusHCAL_slopeECAL_ = 1.05;
  paramECALplusHCAL_slopeHCAL_ = 1.06;
  paramECALplusHCAL_offset_ = 6.11;
}


PFEnergyCalibration::PFEnergyCalibration( double e_slope  , 
					  double e_offset , 
					  double eh_eslope,
					  double eh_hslope,
					  double eh_offset,
					  double h_slope  ,
					  double h_offset ,
					  double h_damping ):
  
  paramECAL_slope_(e_slope),
  paramECAL_offset_(e_offset),
  paramECALplusHCAL_slopeECAL_(eh_eslope),
  paramECALplusHCAL_slopeHCAL_(eh_hslope),
  paramECALplusHCAL_offset_(eh_offset),
  paramHCAL_slope_(h_slope),
  paramHCAL_offset_(h_offset),
  paramHCAL_damping_(h_damping) {}






void PFEnergyCalibration::setCalibrationParametersEm(double paramECAL_slope, 
						     double paramECAL_offset) {

//--- set calibration parameters for energy deposits of electrons 
//    and photons in ECAL;
//    this member function is needed by PFRootEvent

  paramECAL_slope_ = paramECAL_slope;
  paramECAL_offset_ = paramECAL_offset;
}



PFEnergyCalibration::~PFEnergyCalibration()
{
//--- nothing to be done yet  
}


double 
PFEnergyCalibration::energyEm(double uncalibratedEnergyECAL, 
			      double eta, double phi) const {

  //--- apply calibration correction
  //    for energy deposits of electrons and photons in ECAL
  //    (eta and phi dependence not implemented yet)
  
  double calibrated = paramECAL_slope_*uncalibratedEnergyECAL;
  calibrated += paramECAL_offset_;

  return calibrated;
}


double 
PFEnergyCalibration::energyHad(double uncalibratedEnergyHCAL, 
			       double eta, double phi) const {
  
  //--- apply calibration correction
  //    for energy deposits of hadrons in HCAL
  //    (eta and phi dependence not implemented yet)
  
  double numerator = paramHCAL_slope_*uncalibratedEnergyHCAL;
  numerator += paramHCAL_offset_;

  double denominator = 1 + exp(paramHCAL_damping_/uncalibratedEnergyHCAL);


  return numerator/denominator;
}


double 
PFEnergyCalibration::energyEmHad(double uncalibratedEnergyECAL, 
				 double uncalibratedEnergyHCAL, 
				 double eta, double phi) const {
//--- apply calibration correction
//    for energy deposits of hadrons in ECAL and HCAL
//    (eta and phi dependence not implemented yet)

  double calibrated = paramECALplusHCAL_slopeECAL_*uncalibratedEnergyECAL;
  calibrated += paramECALplusHCAL_slopeHCAL_*uncalibratedEnergyHCAL;
  calibrated += paramECALplusHCAL_offset_;

  return calibrated;
}
  
std::ostream& operator<<(std::ostream& out, 
			 const PFEnergyCalibration& calib) {

  if(!out ) return out;

  out<<"PFEnergyCalibration -- "<<endl;
  out<<"ecal      = "<<calib.paramECAL_slope_
     <<" x E + "<< calib.paramECAL_offset_<<endl;
  out<<"hcal only = <add formula>"
     <<calib.paramHCAL_slope_<<","
     <<calib.paramHCAL_offset_<<","
     <<calib.paramHCAL_damping_<<endl;
  out<<"ecal+hcal = "<<calib.paramECALplusHCAL_slopeECAL_<<" x E_e + "
     <<calib.paramECALplusHCAL_slopeHCAL_<<" x E_h + "
     <<calib.paramECALplusHCAL_offset_<<endl;

  return out;
}
