#ifndef RecoParticleFlow_PFClusterTools_PFSCEnergyCalibration_h
#define RecoParticleFlow_PFClusterTools_PFSCEnergyCalibration_h 
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include <iostream>


// -*- C++ -*-
//
// Package:    PFClusterTools
// Class:      PFSuperClusterEnergyCalibration
// 
/**\class

 Description: An auxiliary class of the Particle-Flow algorithm,
              for the Calibration of the electron SuperCluster 

 Implementation:
     Original Implementation of Calibration functions by Daniele Benedetti

 
*/
//
// Original Author:  Daniele Benedetti
//         Created:  Fri Dec  4 10:18:18 CDT 2006




class PFSCEnergyCalibration 
{
 public:
  PFSCEnergyCalibration(); // default constructor;
                           // needed by PFRootEvent
  
  PFSCEnergyCalibration(std::vector<double> &barrelCorr,
			std::vector<double> &endcapCorr);
  
  
  ~PFSCEnergyCalibration();

  // ecal calibration
  double SCCorrEtEtaBarrel(double et, double eta);
  double SCCorrEtEtaEndcap(double et, double eta);
  

 private:
  std::vector<double> barrelCorr_;
  std::vector<double> endcapCorr_;
  double cc[9];
  double bb[11];

};

#endif


