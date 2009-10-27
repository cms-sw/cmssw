#ifndef RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h
#define RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h 

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class TF1;

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

 Modification 
     To include energy-dependent and angular-dependent calibration
     By Patrick Janot
 
*/
//
// Original Author:  Christian Veelken
//         Created:  Tue Aug  8 16:26:18 CDT 2006
// $Id: PFEnergyCalibration.h,v 1.10 2009/05/19 18:59:33 pjanot Exp $
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
		       double h_damping,
		       unsigned newCalib = 0);

  ~PFEnergyCalibration();

  // ecal calibration
  double energyEm(double uncalibratedEnergyECAL, 
		  double eta=0, double phi=0) const;
  
  double energyEm(const reco::PFCluster& clusterEcal,
		  std::vector<double> &EclustersPS1,
		  std::vector<double> &EclustersPS2,
		  bool crackCorrection = true);

  double energyEm(const reco::PFCluster& clusterEcal,
		  std::vector<double> &EclustersPS1,
		  std::vector<double> &EclustersPS2,
		  double &ps1,double&ps2,
		  bool crackCorrection=true);

  // HCAL only calibration
  double energyHad(double uncalibratedEnergyHCAL, 
		   double eta=0, double phi=0) const;
  
  
  // ECAL+HCAL (abc) calibration
  double energyEmHad(double uncalibratedEnergyECAL, 
		     double uncalibratedEnergyHCAL, 
		     double eta=0, double phi=0) const;

  // ECAL+HCAL (abc-alpha-beta) calibration, with E and eta dependent coefficients
  void energyEmHad(double t, double& e, double&h, double eta, double phi) const;
  
  // set calibration parameters for energy deposits of electrons and photons in ECAL; this member function is needed by PFRootEvent
  void setCalibrationParametersEm(double paramECAL_slope, 
				  double paramECAL_offset);

  // Initialize E- and eta-dependent coefficient functional form
  void initializeCalibrationFunctions();

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

 private:
  
  double minimum(double a,double b);
  double dCrackPhi(double phi, double eta);
  double CorrPhi(double phi, double eta);
  double CorrEta(double eta);
  double CorrBarrel(double E, double eta);
  double Alpha(double eta);
  double Beta(double E, double eta);
  double Gamma(double etaEcal);
  double EcorrBarrel(double E, double eta, double phi, bool crackCorrection=true);
  double EcorrZoneBeforePS(double E, double eta);
  double EcorrPS(double eEcal,double ePS1,double ePS2,double etaEcal);
  double EcorrPS(double eEcal,double ePS1,double ePS2,double etaEcal,double&, double&);
  double EcorrPS_ePSNil(double eEcal,double eta);
  double EcorrZoneAfterPS(double E, double eta);
  double Ecorr(double eEcal,double ePS1,double ePS2,double eta,double phi,bool crackCorrection=true);
  double Ecorr(double eEcal,double ePS1,double ePS2,double eta,double phi,double&,double&,bool crackCorrection=true);

  // Barrel calibration (eta 0.00 -> 1.48)
  TF1* faBarrel;
  TF1* fbBarrel; 
  TF1* fcBarrel; 
  TF1* faEtaBarrel; 
  TF1* fbEtaBarrel; 

  // Endcap calibration (eta 1.48 -> 3.xx)
  TF1* faEndcap;
  TF1* fbEndcap; 
  TF1* fcEndcap; 
  TF1* faEtaEndcap; 
  TF1* fbEtaEndcap; 

  // Threshold correction (offset)
  double threshE, threshH;

};

#endif


