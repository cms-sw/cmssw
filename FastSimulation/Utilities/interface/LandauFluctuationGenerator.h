#ifndef FastSimulation_Utilities_LandauFluctuationGenerator_H
#define FastSimulation_Utilities_LandauFluctuationGenerator_H

#include "FastSimulation/Utilities/interface/BaseNumericalRandomGenerator.h"

#include <cmath>

/** Numerical Random Generator for Landau Fluctuations.
 * The constructor integrates and inverses the Ersaztz for the 
 * Landau fluctuation density probability parametrization, and 
 * the method landau() randomly a number according to this 
 * density probability
 * 
 * \author Patrick Janot, CERN
 * $Date 8-Jan-2004
 */

class RandomEngine;

class LandauFluctuationGenerator : public BaseNumericalRandomGenerator
{
 public:

  /// Constructor : initialization of the Random Generator
  LandauFluctuationGenerator(const RandomEngine* engine) : 
    BaseNumericalRandomGenerator(engine,-3.5,25.) {
    initialize();
  }

  /// Default destructor
  virtual ~LandauFluctuationGenerator() {}

  /// Random generator of the dE/dX spread (Landau function)  
  double landau() const { return generate(); }
  
  /// The probability density function implementation
  virtual double function(double x) { return ersatzt(x); }

 private:

  /// Ersatzt for Landau Fluctuations (very good approximation)
  double ersatzt(double x) { 
    return  std::exp(-0.5 * ( x + std::exp(-x) )) / std::sqrt (2. *M_PI); 
  }
};
#endif
