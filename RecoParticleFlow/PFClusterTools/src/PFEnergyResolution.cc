#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

#include <math.h>
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

/*
double PFEnergyResolution::getEnergyResolutionEm(double energyECAL, double eta, double phi) const
{
//--- not implemented yet
}
*/

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
