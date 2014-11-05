/*                                                                           
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru
November. 2, 2005                                

*/

#include "TLorentzVector.h"
#include "TVector3.h"

#include "GeneratorInterface/Hydjet2Interface/interface/Particle.h"
#include "GeneratorInterface/Hydjet2Interface/interface/UKUtility.h"
 
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

extern CLHEP::HepRandomEngine* hjRandomEngine;

const double GeV = 1.;
const double fermi = 1.;
const double hbarc = 0.197 * GeV * fermi; 
const double w = 1.0 / 0.1973;
const double hbarc_squared = hbarc * hbarc;

void IsotropicR3(double r, double *x, double *y, double *z) {
  double pZ  = 1. - 2.*(CLHEP::RandFlat::shoot(hjRandomEngine));
  double st  = TMath::Sqrt(1. - pZ * pZ) * r;
  double phi = 2. * TMath::Pi() * (CLHEP::RandFlat::shoot(hjRandomEngine));

  *x = st * cos(phi);
  *y = st * sin(phi);
  *z = pZ * r;
}

void IsotropicR3(double r, TVector3 &pos) {
  double pZ  = 1. - 2.* (CLHEP::RandFlat::shoot(hjRandomEngine));  
  double st  = TMath::Sqrt(1. - pZ * pZ) * r;
  double phi = 2. * TMath::Pi() * (CLHEP::RandFlat::shoot(hjRandomEngine));
  pos.SetX(st * TMath::Cos(phi));
  pos.SetY(st * TMath::Sin(phi));
  pos.SetZ(pZ * r);
}

void MomAntiMom(TLorentzVector &mom, double mass, TLorentzVector &antiMom, 
		double antiMass, double initialMass) {
  double r = initialMass * initialMass - mass * mass - antiMass * antiMass;
  if (r * r - 4 * mass * mass * antiMass * antiMass < 0.) throw "MomAntiMom";
      
  double pAbs = .5 * TMath::Sqrt(r * r - 4 * mass * mass * antiMass * antiMass) / initialMass;
  TVector3 mom3;
  IsotropicR3(pAbs, mom3);
  mom.SetVectM(mom3, mass);
  antiMom.SetVectM(- mom3, antiMass);
}


