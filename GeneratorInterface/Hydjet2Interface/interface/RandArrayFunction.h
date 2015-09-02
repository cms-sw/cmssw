#ifndef RANDARRAYFUNCTION_INCLUDED
#define RANDARRAYFUNCTION_INCLUDED

#include <vector>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
extern CLHEP::HepRandomEngine* hjRandomEngine;
/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/
//This class is taken from the GEANT4 tool kit and changed!!!!!

//========================================================================================
//RandArrayFunction defines several methods for shooting generally distributed random values, 
//given a user-defined probability distribution function.

//The probability distribution function Pdf must be provided by the user as an array of 
//positive real numbers. The array size must also be provided. Pdf doesn't need to be 
//normalized to 1.

// if IntType = 0 ( default value ) a uniform random number is
// generated using the StandardRand() engine. The uniform number is then transformed
// to the user's distribution using the cumulative probability
// distribution constructed from his histogram. The cumulative
// distribution is inverted using a binary search for the nearest
// bin boundary and a linear interpolation within the
// bin. RandArrayFunction therefore generates a constant density within
// each bin.
// if IntType = 1 no interpolation is performed and the result is a
// discrete distribution.

//A speculate set of Shoot()/ShootArray() and Fire()/FireArray() methods is provided 
//to Shoot random numbers via an instantiated RandArrayFunction object. These methods 
//act directly on the flat distribution provided by a StandardRand() engine. 
//An Operator () is also provided. 

//  example.
//      ...
//      double* Pdf;
//      int fNBins;
//      ...
//      RandArrayFunction FunctDist(Pdf,fNBins);
//      ... 
//      double num = FunctDist.Shoot();//Shoot() provides the same functionality as Fire()

//  example.
//      ...
//      double* Pdf;
//      int fNBins;
//      ...
//      RandArrayFunction FunctDist(Pdf,fNBins);
//      ... 
//      double num = FunctDist(); 

//  example.
//      ...
//      double* Pdf;
//      int fNBins;
//      ...
//      RandArrayFunction FunctDist(Pdf,fNBins);
//      ...
//	    int size = 50;
//	    double* vect = new double[size];
//      FunctDist.FireArray (size, vect);

//========================================================================================

class RandArrayFunction {
 private:
  std::vector<double> fIntegralPdf;
  int                 fNBins;
  double              fOneOverNbins;
  int                 fInterpolationType;

 public:
  RandArrayFunction(const double *aProbFunc, int theProbSize, int interpolationType = 0);
  RandArrayFunction(int probSize, int interpolationType = 0);

  double Shoot()const;
  double Fire()const;
  double operator()()const;
  void     ShootArray(int size, double *array)const;
  void     FireArray(int size, double *array)const;

  void     PrepareTable(const double *aProbFunc);

 private:
  void     UseFlatDistribution();
  double MapRandom(double rand)const;
  double StandardRand()const;
};

inline double RandArrayFunction::StandardRand() const {
  return CLHEP::RandFlat::shoot(hjRandomEngine);
}

inline double RandArrayFunction::Fire() const {
  return MapRandom(StandardRand());
}

inline double RandArrayFunction::Shoot() const {
  return Fire();
}

inline double RandArrayFunction::operator()() const {
  return Fire();
}

inline void RandArrayFunction::ShootArray(int size, double *array) const {
  FireArray(size, array);
}

#endif
