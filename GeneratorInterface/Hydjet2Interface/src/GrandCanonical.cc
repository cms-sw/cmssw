/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/
#include <iostream>
#include <TMath.h>

#include "GeneratorInterface/Hydjet2Interface/interface/GrandCanonical.h"
#include "GeneratorInterface/Hydjet2Interface/interface/HankelFunction.h"
#include "GeneratorInterface/Hydjet2Interface/interface/UKUtility.h"

GrandCanonical::GrandCanonical() {
  fInitialized = kFALSE;
  fNMax = -1111;
  fTemperature = -1111;
  fBaryonPotential = -1111;
  fStrangePotential = -1111;
  fElectroPotential = -1111;
  fCharmPotential = -1111;
}

GrandCanonical::GrandCanonical(int nmax, double temperature, double baryonPotential, double strangePotential, double electroPotential, double charmPotential) {
  fNMax = nmax;
  fTemperature = temperature;
  fBaryonPotential = baryonPotential;
  fStrangePotential = strangePotential;
  fElectroPotential = electroPotential;
  fCharmPotential = charmPotential;
  fInitialized = kTRUE;
}

GrandCanonical::~GrandCanonical() {}

void GrandCanonical::Temperature(double value) {
  fTemperature = value;
  if(fNMax!=-1111 && fBaryonPotential!=-1111 && fStrangePotential!=-1111 && fElectroPotential!=-1111 && fCharmPotential!=-1111)
    fInitialized = kTRUE;
}

void GrandCanonical::BaryonPotential(double value) {
  fBaryonPotential = value;
  if(fNMax!=-1111 && fTemperature!=-1111 && fStrangePotential!=-1111 && fElectroPotential!=-1111 && fCharmPotential!=-1111)
    fInitialized = kTRUE;
}

void GrandCanonical::StrangePotential(double value) {
  fStrangePotential = value;
  if(fNMax!=-1111 && fTemperature!=-1111 && fBaryonPotential!=-1111 && fElectroPotential!=-1111 && fCharmPotential!=-1111)
    fInitialized = kTRUE;
}

void GrandCanonical::ElectroPotential(double value) {
  fElectroPotential = value;
  if(fNMax!=-1111 && fTemperature!=-1111 && fBaryonPotential!=-1111 && fStrangePotential!=-1111 && fCharmPotential!=-1111)
    fInitialized = kTRUE;
}

void GrandCanonical::CharmPotential(double value) {
  fCharmPotential = value;
  if(fNMax!=-1111 && fTemperature!=-1111 && fBaryonPotential!=-1111 && fStrangePotential!=-1111 && fElectroPotential!=-1111)
    fInitialized = kTRUE;
}

void GrandCanonical::NMax(int value) {
  fNMax = value;
  if(fTemperature!=-1111 && fBaryonPotential!=-1111 && fStrangePotential!=-1111 && fElectroPotential!=-1111 && fCharmPotential!=-1111)
    fInitialized = kTRUE;
}


double GrandCanonical::ParticleEnergyDensity(ParticlePDG* particle) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::ParticleEnergyDensity", "GrandCanonical object not fully initialized!!");
  
  // Compute the particle energy density
  double degFactor = 2.*particle->GetSpin() + 1.;                                    // degeneracy factor
  double mass = particle->GetMass();                                                // PDG table mass
  double d = int(2.*particle->GetSpin()) & 1 ? 1. : -1;                            // Bose-Einstein/Fermi-Dirac factor
  double preFactor = (degFactor*mass*mass*fTemperature*fTemperature/hbarc/hbarc/hbarc)/(2.*TMath::Pi()*TMath::Pi()); 

  double postFactor = 0.;
  //compute chemical potential
  double potential = fBaryonPotential * particle->GetBaryonNumber() + 
    fStrangePotential * particle->GetStrangeness() +
    fElectroPotential * particle->GetElectricCharge() +
    fCharmPotential * particle->GetCharmness();

  for(int n = 1; n <= fNMax; ++n) {
    postFactor += TMath::Power(-d, n+1)/(n*n) *
      TMath::Exp(n*potential/fTemperature) *
      (3.*HankelKn(2, n*mass/fTemperature) + n*mass/fTemperature*HankelK1(n*mass/fTemperature));
  }
  return preFactor * postFactor;
}

double GrandCanonical::ParticleNumberDensity(ParticlePDG* particle) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::ParticleNumberDensity", "GrandCanonical object not fully initialized!!");

  double degFactor = 2.*particle->GetSpin() + 1.;
  double mass = particle->GetMass();     
  double d = int(2*particle->GetSpin()) & 1 ? 1. : -1.;
  double preFactor = (degFactor*mass*mass*fTemperature/hbarc/hbarc/hbarc)/(2.*TMath::Pi()*TMath::Pi());

  double postFactor = 0.;
  double potential = fBaryonPotential * particle->GetBaryonNumber() + 
    fStrangePotential * particle->GetStrangeness() +
    fElectroPotential * particle->GetElectricCharge() +
    fCharmPotential * particle->GetCharmness();
                           
  for(int n = 1; n <= fNMax; ++n) {
    postFactor += TMath::Power(-d, n+1)/n * 
      TMath::Exp(n*potential/fTemperature) *
      HankelKn(2, n*mass/fTemperature);        
  }
  return preFactor * postFactor;
}


double GrandCanonical::EnergyDensity(DatabasePDG* database) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::EnergyDensity", "GrandCanonical object not fully initialized!!");

  double meanEnergyDensity = 0.;

  for(int currParticle = 0; currParticle<database->GetNParticles(); currParticle++) {
    ParticlePDG *particle = database->GetPDGParticleByIndex(currParticle);
    meanEnergyDensity += ParticleEnergyDensity(particle);
  }

  return meanEnergyDensity;
}

double GrandCanonical::BaryonDensity(DatabasePDG* database) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::BaryonDensity", "GrandCanonical object not fully initialized!!");

  double meanBaryonDensity = 0.;

  for(int currParticle = 0; currParticle<database->GetNParticles(); currParticle++) {
    ParticlePDG *particle = database->GetPDGParticleByIndex(currParticle);
    meanBaryonDensity += ParticleNumberDensity(particle)*particle->GetBaryonNumber();
  }
  return meanBaryonDensity;
}

double GrandCanonical::StrangeDensity(DatabasePDG* database) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::StrangeDensity", "GrandCanonical object not fully initialized!!");

  double meanStrangeDensity = 0.;

  for(int currParticle = 0; currParticle<database->GetNParticles(); currParticle++) {
    ParticlePDG *particle = database->GetPDGParticleByIndex(currParticle);
    meanStrangeDensity += ParticleNumberDensity(particle)*particle->GetStrangeness();
  }

  return meanStrangeDensity;
}

double GrandCanonical::ElectroDensity(DatabasePDG* database) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::ElectroDensity", "GrandCanonical object not fully initialized!!");

  double meanElectroDensity = 0.;
  
  //hadrons
  for(int currParticle = 0; currParticle<database->GetNParticles(); currParticle++) {
    ParticlePDG *particle = database->GetPDGParticleByIndex(currParticle);
    meanElectroDensity += ParticleNumberDensity(particle)*particle->GetElectricCharge();
  }

  return meanElectroDensity;
}

double GrandCanonical::CharmDensity(DatabasePDG* database) {
  // Check if all the thermodinamic parameters are set
  if(!fInitialized)
    Fatal("GrandCanonical::CharmDensity", "GrandCanonical object not fully initialized!!");

  double meanCharmDensity = 0.;

  for(int currParticle = 0; currParticle<database->GetNParticles(); currParticle++) {
    ParticlePDG *particle = database->GetPDGParticleByIndex(currParticle);
    meanCharmDensity += ParticleNumberDensity(particle)*particle->GetCharmness();
  }

  return meanCharmDensity;
}
