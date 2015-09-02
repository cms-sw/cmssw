/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#ifndef GRANDCANONICAL_INCLUDED
#define GRANDCANONICAL_INCLUDED
#include "ParticlePDG.h"
#include "DatabasePDG.h"

class GrandCanonical {

 private:

  double    fTemperature;     
  double    fBaryonPotential;	
  double    fStrangePotential;
  double    fElectroPotential;
  double    fCharmPotential;

  //  Number of terms for summation, if fNMax = 1 then 
  //  Maxwell-Boltzmann distribution will be recovered
  int       fNMax;
  bool fInitialized;

 public:
  GrandCanonical();
  GrandCanonical(int nmax, double temperature, double baryonPotential, double strangePotential, double electroPotential, double charmPotential);
  ~GrandCanonical();

  void     Temperature(double value); 
  double Temperature() { return fTemperature; }
  void     BaryonPotential(double value);
  double BaryonPotential() { return fBaryonPotential; }
  void     StrangePotential(double value);
  double StrangePotential() { return fStrangePotential; }
  void     ElectroPotential(double value);
  double ElectroPotential() { return fElectroPotential; }
  void     CharmPotential(double value);
  double CharmPotential() { return fCharmPotential; }

  void     NMax(int value); 
  int    NMax() { return fNMax; }

  // compute of system baryon number, system strangeness, system charge and 
  // system energy
  // calculate system energy density
  double EnergyDensity(DatabasePDG* database);
  // calculate system baryon density
  double BaryonDensity(DatabasePDG* database);
  // calculate system strangeness density
  double StrangeDensity(DatabasePDG* database);
  // calculate system electro density
  double ElectroDensity(DatabasePDG* database);
  // compute of particle number density 
  double CharmDensity(DatabasePDG* database);

  // compute of particle number density
  double ParticleNumberDensity(ParticlePDG* particle);
  // compute the particle energy density 
  double ParticleEnergyDensity(ParticlePDG* particle); 
};

#endif
