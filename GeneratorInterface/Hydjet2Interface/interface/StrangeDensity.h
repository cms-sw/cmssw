/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

//This class is used to obtain grand canonical description  of strange density 
//by means of the temperature and chemical potentials (input). As for output 
//we get  strange density.

#ifndef NAStrangeDensity_h
#define NAStrangeDensity_h 1

#include "MathUtil.h"

#include "HankelFunction.h"
#include "Particle.h"
#include "DatabasePDG.h"
#include "ParticlePDG.h"

class NAStrangeDensity {

 private:
  //input
  double fTemperature;
  double fBaryonPotential;	
  double fStrangePotential;
  int fNMax;   //number of terms for summation, if nMax = 1 then
  //Maxwell-Boltzmann distribution will be recovered	

  double ParticleNumberDensity(ParticlePDG* particle);

 public:
  NAStrangeDensity();
  ~NAStrangeDensity(){};

  //for input
  void SetTemperature(double value) {fTemperature = value;}
  void SetBaryonPotential(double value) {fBaryonPotential = value;}
  void SetStrangePotential(double value) {fStrangePotential = value;}
  void SetNMax(int value) {
    fNMax = value; 
    if(fNMax < 1) fNMax = 1;
  }
  // compute hadron system strangeness density
  double StrangenessDensity(DatabasePDG* database);
};

#endif
