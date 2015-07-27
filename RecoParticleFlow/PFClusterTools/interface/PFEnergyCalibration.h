#ifndef RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h
#define RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h 

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"

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
//
//

#include <iostream>

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFEnergyCalibration 
{
 public:
  PFEnergyCalibration(); 

  ~PFEnergyCalibration();

  // ecal calibration for photons
  double energyEm(const reco::PFCluster& clusterEcal,
		  std::vector<double> &EclustersPS1,
		  std::vector<double> &EclustersPS2,
		  bool crackCorrection = true) const;
  double energyEm(const reco::PFCluster& clusterEcal,
		  double ePS1,  double ePS2,
		  bool crackCorrection = true) const;

  double energyEm(const reco::PFCluster& clusterEcal,
		  std::vector<double> &EclustersPS1,
		  std::vector<double> &EclustersPS2,
		  double &ps1,double&ps2,
		  bool crackCorrection=true) const;
  double energyEm(const reco::PFCluster& clusterEcal,
		  double ePS1, double ePS2,
		  double &ps1,double&ps2,
		  bool crackCorrection=true) const;

  // ECAL+HCAL (abc) calibration, with E and eta dependent coefficients, for hadrons
  void energyEmHad(double t, double& e, double&h, double eta, double phi) const;
  
  // Initialize default E- and eta-dependent coefficient functional form
  void initializeCalibrationFunctions();

  // Set the run-dependent calibration functions from the global tag
  void setCalibrationFunctions(const PerformancePayloadFromTFormula *thePFCal) {
    pfCalibrations = thePFCal;
  }

  friend std::ostream& operator<<(std::ostream& out, 
				  const PFEnergyCalibration& calib);

 protected:

  // Calibration functions from global tag
  const PerformancePayloadFromTFormula *pfCalibrations;
  
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

 private:
  
  double minimum(double a,double b) const;
  double dCrackPhi(double phi, double eta) const;
  double CorrPhi(double phi, double eta) const;
  double CorrEta(double eta) const;
  double CorrBarrel(double E, double eta) const;
  double Alpha(double eta) const;
  double Beta(double E, double eta) const;
  double Gamma(double etaEcal) const;
  double EcorrBarrel(double E, double eta, double phi, bool crackCorrection=true) const;
  double EcorrZoneBeforePS(double E, double eta) const;
  double EcorrPS(double eEcal,double ePS1,double ePS2,double etaEcal) const;
  double EcorrPS(double eEcal,double ePS1,double ePS2,double etaEcal,double&, double&) const;
  double EcorrPS_ePSNil(double eEcal,double eta) const;
  double EcorrZoneAfterPS(double E, double eta) const;
  double Ecorr(double eEcal,double ePS1,double ePS2,double eta,double phi,bool crackCorrection=true) const;
  double Ecorr(double eEcal,double ePS1,double ePS2,double eta,double phi,double&,double&,bool crackCorrection=true) const;

  // The calibration functions
  double aBarrel(double x) const;
  double bBarrel(double x) const;
  double cBarrel(double x) const;
  double aEtaBarrel(double x) const;
  double bEtaBarrel(double x) const; 
  double aEndcap(double x) const;
  double bEndcap(double x) const;
  double cEndcap(double x) const;
  double aEtaEndcap(double x) const; 
  double bEtaEndcap(double x) const;

  // Threshold correction (offset)
  double threshE, threshH;

};

#endif


