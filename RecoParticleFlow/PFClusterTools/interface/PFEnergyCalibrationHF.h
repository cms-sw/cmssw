#ifndef RecoParticleFlow_PFClusterTools_PFEnergyCalibrationHF_h
#define RecoParticleFlow_PFClusterTools_PFEnergyCalibrationHF_h 

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

// -*- C++ -*-
//
// Package:    PFClusterTools
// Class:      PFEnergyCalibrationHF
// 
/**\class

 Description: An auxiliary class of the Particle-Flow algorithm,
              for the Calibration of Energy Deposits HF.

 Modified version of PFEnergyCalibration for the HF calibration.
     By Auguste Besson

*/

#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFEnergyCalibrationHF 
{
 public:
  PFEnergyCalibrationHF(); // default constructor;
                         // needed by PFRootEvent
 explicit PFEnergyCalibrationHF(bool calibHF_use, 
					     const std::vector<double>& calibHF_eta_step,
					     const std::vector<double>& calibHF_a_EMonly,
					     const std::vector<double>& calibHF_b_HADonly,
					     const std::vector<double>& calibHF_a_EMHAD,
					     const std::vector<double>& calibHF_b_EMHAD);
  
  ~PFEnergyCalibrationHF();


  double energyEm(double uncalibratedEnergyECAL, 
		  double eta, double phi);

  // HCAL only calibration
  double energyHad(double uncalibratedEnergyHCAL, 
		   double eta, double phi) ;
  
  
  // ECAL+HCAL (abc) calibration
  double energyEmHad(double uncalibratedEnergyECAL, 
		     double uncalibratedEnergyHCAL, 
		     double eta, double phi) ;


  
  friend std::ostream& operator<<(std::ostream& out, 
  				  const PFEnergyCalibrationHF& calib);

  const bool& getcalibHF_use() const
    { return calibHF_use_;}
  const std::vector<double>& getcalibHF_eta_step() const
    {return calibHF_eta_step_;}
  const std::vector<double>& getcalibHF_a_EMonly() const
    {return calibHF_a_EMonly_;}
  const std::vector<double>& getcalibHF_b_HADonly() const
    {return calibHF_b_HADonly_;}
  const std::vector<double>& getcalibHF_a_EMHAD() const
    {return calibHF_a_EMHAD_;}
  const std::vector<double>& getcalibHF_b_EMHAD() const
    {return calibHF_b_EMHAD_;}


 protected:

 private:
  bool calibHF_use_;
  std::vector<double>  calibHF_eta_step_;
  std::vector<double>  calibHF_a_EMonly_;
  std::vector<double>  calibHF_b_HADonly_;
  std::vector<double>  calibHF_a_EMHAD_;
  std::vector<double>  calibHF_b_EMHAD_ ;
};

#endif


