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

class RandomEngineAndDistribution;

class LandauFluctuationGenerator : public BaseNumericalRandomGenerator
{
 public:

  /// Constructor : initialization of the Random Generator
  LandauFluctuationGenerator() :
    BaseNumericalRandomGenerator(-3.5,25.) {
    initialize();
  }

  /// Default destructor
  ~LandauFluctuationGenerator() override {}

  /// Random generator of the dE/dX spread (Landau function)  
  double landau(RandomEngineAndDistribution const* random) const { return generate(random); }
  
  /// The probability density function implementation
  double function(double x) override { return ersatzt(x); }

 private:

  /// Ersatzt for Landau Fluctuations (very good approximation)
  double ersatzt(double x) { 
    return  std::exp(-0.5 * ( x + std::exp(-x) )) / std::sqrt (2. *M_PI); 
  }
};
#endif
