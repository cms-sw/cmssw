#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

PFEnergyCalibration::PFEnergyCalibration()
{
//--- initialize calibration parameters
//    for energy correction applied to energy deposits of electrons and photons in ECAL
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
  paramECALplusHCAL_offset_ = 8.61;
}

PFEnergyCalibration::PFEnergyCalibration(const edm::ParameterSet& parameters)
{
//--- initialize calibration parameters
//    for energy correction applied to energy deposits of electrons and photons in ECAL
  paramECAL_slope_ = parameters.getParameter<double>("pf_ECAL_calib_p1");
  paramECAL_offset_ = parameters.getParameter<double>("pf_ECAL_calib_p0");
  
//-------------------------------------------------------------------------------
// calibration parameters for energy deposits of hadrons not implemented yet,
// so use defaults
//-------------------------------------------------------------------------------

//--- initialize calibration parameters
//    for energy correction applied to energy deposits of hadrons in HCAL
  paramHCAL_slope_ = 2.17;
  paramHCAL_offset_ = 1.73;
  paramHCAL_damping_ = 2.49;

//--- initialize calibration parameters
//    for energy correction applied to combined energy deposits of hadrons in HCAL and ECAL
  paramECALplusHCAL_slopeECAL_ = 1.05;
  paramECALplusHCAL_slopeHCAL_ = 1.06;
  paramECALplusHCAL_offset_ = 8.61;
}

void PFEnergyCalibration::setCalibrationParametersEm(double paramECAL_slope, double paramECAL_offset)
{
//--- set calibration parameters for energy deposits of electrons and photons in ECAL;
//    this member function is needed by PFRootEvent

  paramECAL_slope_ = paramECAL_slope;
  paramECAL_offset_ = paramECAL_offset;
}

PFEnergyCalibration::~PFEnergyCalibration()
{
//--- nothing to be done yet  
}

double PFEnergyCalibration::getCalibratedEnergyEm(double uncalibratedEnergyECAL, double eta, double phi) const
{
//--- apply calibration correction
//    for energy deposits of electrons and photons in ECAL
//    (eta and phi dependence not implemented yet)

  return paramECAL_slope_*uncalibratedEnergyECAL + paramECAL_offset_;
}

double PFEnergyCalibration::getCalibratedEnergyHad(double uncalibratedEnergyHCAL, double eta, double phi) const
{
//--- apply calibration correction
//    for energy deposits of hadrons in HCAL
//    (eta and phi dependence not implemented yet)

  return (paramHCAL_slope_*uncalibratedEnergyHCAL + paramHCAL_offset_)/(1 + exp(paramHCAL_damping_/uncalibratedEnergyHCAL));
}

double PFEnergyCalibration::getCalibratedEnergyHad(double uncalibratedEnergyECAL, double uncalibratedEnergyHCAL, double eta, double phi) const
{
//--- apply calibration correction
//    for energy deposits of hadrons in ECAL and HCAL
//    (eta and phi dependence not implemented yet)

  return paramECALplusHCAL_slopeECAL_*uncalibratedEnergyECAL + paramECALplusHCAL_slopeHCAL_*uncalibratedEnergyHCAL + paramECALplusHCAL_offset_;
}
