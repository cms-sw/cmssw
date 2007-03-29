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
     Original Implementation of Calibration functions in PFAlgo/PFBlock by Colin Bernet;
     Code moved into separate Class by Christian Veelken 
*/
//
// Original Author:  Christian Veelken
//         Created:  Tue Aug  8 16:26:18 CDT 2006
// $Id: PFEnergyCalibration.h,v 1.1 2007/03/27 15:13:53 veelken Exp $
//
//

#include <iostream>

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFEnergyCalibration 
{
 public:
  PFEnergyCalibration(); // default constructor;
                         // needed by PFRootEvent
  //PFEnergyCalibration(const edm::ParameterSet& parameters);
  ~PFEnergyCalibration();

  double getCalibratedEnergyEm(double uncalibratedEnergyECAL, double eta, double phi) const;
  double getCalibratedEnergyHad(double uncalibratedEnergyHCAL, double eta, double phi) const;
  double getCalibratedEnergyHad(double uncalibratedEnergyECAL, double uncalibratedEnergyHCAL, double eta, double phi) const;
  
  void setCalibrationParametersEm(double paramECAL_slope, double paramECAL_offset); // set calibration parameters for energy deposits of electrons and photons in ECAL;
                                                                                    // this member function is needed by PFRootEvent

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


