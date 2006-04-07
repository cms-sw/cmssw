#ifndef LandauFluctuationGenerator_H
#define LandauFluctuationGenerator_H

#include "FastSimulation/Utilities/interface/BaseNumericalRandomGenerator.h"


/** Numerical Random Generator for Landau Fluctuations.
 * The constructor integrates and inverses the Ersaztz for the 
 * Landau fluctuation density probability parametrization, and 
 * the method landau() randomly a number according to this 
 * density probability
 * 
 * \author Patrick Janot, CERN
 * $Date 8-Jan-2004
 */

class LandauFluctuationGenerator : public BaseNumericalRandomGenerator
{
 public:

  /// Constructor : initialization of the Random Generator
  LandauFluctuationGenerator() : BaseNumericalRandomGenerator(-3.5,25.) {
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
    return  exp(-0.5 * ( x + exp(-x) )) / sqrt (2. *M_PI); 
  }
};
#endif
