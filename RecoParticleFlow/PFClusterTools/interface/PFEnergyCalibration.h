#ifndef RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h
#define RecoParticleFlow_PFClusterTools_PFEnergyCalibration_h 

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"

#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"

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

  void initAlphaGamma_ESplanes_fromDB(const ESEEIntercalibConstants* esEEInterCalib){
    esEEInterCalib_ = esEEInterCalib;
  }


  friend std::ostream& operator<<(std::ostream& out, 
				  const PFEnergyCalibration& calib);

 protected:

  // Calibration functions from global tag
  const PerformancePayloadFromTFormula *pfCalibrations;
  const ESEEIntercalibConstants* esEEInterCalib_;
  
  // Barrel calibration (eta 0.00 -> 1.48)
  std::unique_ptr<TF1> faBarrel;
  std::unique_ptr<TF1> fbBarrel; 
  std::unique_ptr<TF1> fcBarrel; 
  std::unique_ptr<TF1> faEtaBarrelEH; 
  std::unique_ptr<TF1> fbEtaBarrelEH; 
  std::unique_ptr<TF1> faEtaBarrelH; 
  std::unique_ptr<TF1> fbEtaBarrelH; 

  // Endcap calibration (eta 1.48 -> 3.xx)
  std::unique_ptr<TF1> faEndcap;
  std::unique_ptr<TF1> fbEndcap; 
  std::unique_ptr<TF1> fcEndcap; 
  std::unique_ptr<TF1> faEtaEndcapEH; 
  std::unique_ptr<TF1> fbEtaEndcapEH; 
  std::unique_ptr<TF1> faEtaEndcapH; 
  std::unique_ptr<TF1> fbEtaEndcapH; 

  //added by Bhumika on 2 august 2018
  std::unique_ptr<TF1> fcEtaBarrelEH;
  std::unique_ptr<TF1> fcEtaEndcapEH;
  std::unique_ptr<TF1> fdEtaEndcapEH;
  std::unique_ptr<TF1> fcEtaBarrelH;
  std::unique_ptr<TF1> fcEtaEndcapH;
  std::unique_ptr<TF1> fdEtaEndcapH;

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
  double aEtaBarrelEH(double x) const;
  double bEtaBarrelEH(double x) const; 
  double aEtaBarrelH(double x) const;
  double bEtaBarrelH(double x) const; 
  double aEndcap(double x) const;
  double bEndcap(double x) const;
  double cEndcap(double x) const;
  double aEtaEndcapEH(double x) const; 
  double bEtaEndcapEH(double x) const;
  double aEtaEndcapH(double x) const; 
  double bEtaEndcapH(double x) const;
  //added by Bhumika on 3 august 2018
  double cEtaBarrelEH(double x) const;
  double cEtaEndcapEH(double x) const;
  double dEtaEndcapEH(double x) const;
  double cEtaBarrelH(double x) const;
  double cEtaEndcapH(double x) const;
  double dEtaEndcapH(double x) const;


  // Threshold correction (offset)
  double threshE, threshH;

};

#endif


