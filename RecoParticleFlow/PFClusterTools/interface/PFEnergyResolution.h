#ifndef RecoParticleFlow_PFClusterTools_PFEnergyResolution_h
#define RecoParticleFlow_PFClusterTools_PFEnergyResolution_h 

// -*- C++ -*-
//
// Package:    PFClusterTools
// Class:      PFEnergyResolution
// 
/**\class

 Description: An auxiliary class of the Particle-Flow algorithm,
              for the Estimation of the Energy Resolution in ECAL and HCAL;
              the Resolution can be parametrized either as function of raw or as function of calibrated Energy Deposits;
              currently, it is parametrized as function of **calibrated** Energy Deposits

 Implementation:
     Original Implementation of Resolution functions in PFAlgo/PFBlock by Colin Bernet;
     Code moved into separate Class by Christian Veelken 
*/
//
// Original Author:  Christian Veelken
//         Created:  Tue Aug  8 16:26:18 CDT 2006
// $Id: PFEnergyResolution.h,v 1.3 2008/09/04 09:29:43 benedet Exp $
//
//

#include <iostream>

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFEnergyResolution 
{
 public:
  PFEnergyResolution(); // default constructor;
                        // needed by PFRootEvent
  //PFEnergyResolution(const edm::ParameterSet& parameters);
  ~PFEnergyResolution();
  
  double getEnergyResolutionEm(double CorrectedEnergy, double eta) const;  //The resolution must be a function of the corrected energy available in PFEnergyCalibration

  double getEnergyResolutionHad(double energyHCAL, double eta, double phi) const;
  //double getEnergyResolutionHad(double energyECAL, double energyHCAL, double eta, double phi) const;
};

#endif


