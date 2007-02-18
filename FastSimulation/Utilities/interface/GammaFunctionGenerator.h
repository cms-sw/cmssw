#ifndef GammaFunctionGenerator_H
#define GammaFunctionGenerator_H

/**
 * This class provides a gamma function generator (a<12)
 * The threshold works correctly only for integer values of the shape parameter (alpha)
 * \author Florian Beaudette
 * $Date: 28 Jan 2005 19:30 */

// FAMOS headers
#include "FastSimulation/Utilities/interface/GammaNumericalGenerator.h"

// CLHEP
#include "CLHEP/GenericFunctions/IncompleteGamma.hh"

//STL
#include <vector>

class RandomEngine;

class GammaFunctionGenerator
{
 public:

  /// Constructor
  GammaFunctionGenerator(const RandomEngine* engine);

  /// Destructor
  virtual ~GammaFunctionGenerator();
  
  /// shoot along a gamma distribution with shape parameter alpha and scale beta 
  /// values > xmin
  double shoot() const;
  
  /// The parameters must be set before shooting
    void setParameters(double a,double b, double xm);

 private:
  /// values 0<a<1.
  double gammaFrac() const;
  /// integer values
  double gammaInt() const;



 private:

  // The integer numerical functions
  std::vector<GammaNumericalGenerator> theGammas;
  
  // The gamma distribution core coefficients
  std::vector<double> coreCoeff;

  // The gamma distribution core proba
  double coreProba;

  // possibility to store different limits
  std::vector<double> approxLimit;
  
  // boundaries
  double xmin;
  double xmax;
  
  // closest lower integer
  unsigned na;
  // alpha-na
  double frac;
  // alpha function parameters 
  double alpha,beta;
  //  Incomlete Gamma = Int(0,x)[t^(alpha-1)exp(-t)dt]/Gamma(alpha);
  Genfun::IncompleteGamma myIncompleteGamma;

  // some useful integrals 
  std::vector<double> integralToApproxLimit;

  // if xmin>xmax
  bool badRange;

  const RandomEngine* random;

};
#endif
