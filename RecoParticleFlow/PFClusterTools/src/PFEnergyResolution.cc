#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include <TMath.h>
#include <cmath>
/*
PFEnergyResolution::PFEnergyResolution(const edm::ParameterSet& parameters)
{
//--- nothing to be done yet
}
*/
PFEnergyResolution::PFEnergyResolution()
{
//--- nothing to be done yet
}

PFEnergyResolution::~PFEnergyResolution()
{
//--- nothing to be done yet  
}


double PFEnergyResolution::getEnergyResolutionEm(double CorrectedEnergy, double eta) const{

  //The parameters S,N,C has been determined with the Full Sim on CMSSW_2_1_0_pre4. 
  //The resolution must be a function of the corrected energy available in PFEnergyCalibration
  //Jonathan Biteau July 2008

  double C;
  double S;
  double N;
  if(TMath::Abs(eta)<1.48){C=0.35/100; S=5.51/100; N=98./1000.;}
  else{C=0; S=12.8/100; N=440./1000.;} 
  double result = TMath::Sqrt(C*C*CorrectedEnergy*CorrectedEnergy + S*S*CorrectedEnergy + N*N);
  return result; 
}


double PFEnergyResolution::getEnergyResolutionHad(double energyHCAL, double eta, double phi) const
{
//--- estimate **relative** resolution of energy measurement (sigmaE/E)
//    for hadrons in depositing energy in HCAL
//    (eta and phi dependence not implemented yet)

  return 1.49356/sqrt(energyHCAL) + 6.62527e-03*sqrt(energyHCAL) - 6.33966e-02;
}
/*
double PFEnergyResolution::getEnergyResolutionHad(double energyECAL, double energyHCAL, double eta, double phi) const
{
//--- estimate **relative** resolution of energy measurement (sigmaE/E)
//    for hadrons depositing energy in ECAL and HCAL
//    (currently, the resolution function for hadrons 
//     is assumed to be the same in ECAL and HCAL)

  return getEnergyResolutionHad(energyECAL + energyHCAL, theta, phi);
}
*/
