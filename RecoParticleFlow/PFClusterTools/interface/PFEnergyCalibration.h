#ifndef RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h
#define RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h 

// -*- C++ -*-
//
// Package:    PFClusterTools
// Class:      PFEnergyCalibration
// 
/**\class

 Description: An auxiliary class of the Particle-Flow algorithm,
              for the Calibration of Energy Deposits in ECAL and HCAL

 Implementation:
     Original Implementation of Calibration functions in PFAlgo/PFBlock 
     by Colin Bernet;
     Code moved into separate Class by Christian Veelken 
*/
//
// Original Author:  Christian Veelken
//         Created:  Tue Aug  8 16:26:18 CDT 2006
// $Id: PFEnergyCalibration.h,v 1.2 2007/03/29 14:32:44 veelken Exp $
//
//

#include <iostream>

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFEnergyCalibration 
{
 public:
  PFEnergyCalibration(); // default constructor;
                         // needed by PFRootEvent

  PFEnergyCalibration( double e_slope,
		       double e_offset, 
		       double eh_eslope,
		       double eh_hslope,
		       double eh_offset,
		       double h_slope,
		       double h_offset,
		       double h_damping );

  ~PFEnergyCalibration();

  // ecal calibration
  double energyEm(double uncalibratedEnergyECAL, 
		  double eta=0, double phi=0) const;
  
  // HCAL only calibration
  double energyHad(double uncalibratedEnergyHCAL, 
		   double eta=0, double phi=0) const;
  
  
  // ECAL+HCAL (abc) calibration
  double energyEmHad(double uncalibratedEnergyECAL, 
		     double uncalibratedEnergyHCAL, 
		     double eta=0, double phi=0) const;
  
  // set calibration parameters for energy deposits of electrons and photons in ECAL; this member function is needed by PFRootEvent
  void setCalibrationParametersEm(double paramECAL_slope, 
				  double paramECAL_offset);

  double paramECAL_slope() const {return  paramECAL_slope_;} 

  double paramECAL_offset() const {return paramECAL_offset_;} 

  double paramECALplusHCAL_slopeECAL() const {
    return paramECALplusHCAL_slopeECAL_;
  } 

  double paramECALplusHCAL_slopeHCAL() const {
    return paramECALplusHCAL_slopeHCAL_;
  } 

  double paramECALplusHCAL_offset() const {return paramECALplusHCAL_offset_;} 

  double paramHCAL_slope() const {return paramHCAL_slope_;} 
  double paramHCAL_offset() const {return paramHCAL_offset_;} 
  double paramHCAL_damping() const {return paramHCAL_damping_;} 

  
  friend std::ostream& operator<<(std::ostream& out, 
				  const PFEnergyCalibration& calib);

 protected:
  double paramECAL_slope_;
  double paramECAL_offset_;
  
  double paramECALplusHCAL_slopeECAL_;
  double paramECALplusHCAL_slopeHCAL_;
  double paramECALplusHCAL_offset_;
  
  double paramHCAL_slope_;
  double paramHCAL_offset_;
  double paramHCAL_damping_;
};

#endif


