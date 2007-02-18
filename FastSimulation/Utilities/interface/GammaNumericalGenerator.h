#ifndef GammaNumericalGenerator_H
#define GammaNumericalGenerator_H
#include "FastSimulation/Utilities/interface/BaseNumericalRandomGenerator.h"

#include <cmath>
/** Numerical Random Generator for Gamma distribution.
 *  Copy of LandauFluctuations
 */

class RandomEngine;

class GammaNumericalGenerator : public BaseNumericalRandomGenerator
{
 public:

  /// Constructor : initialization of the Random Generator
   GammaNumericalGenerator(const RandomEngine* engine, 
			   double a=0,double b=0,double x1=0,double x2=0) : 
     BaseNumericalRandomGenerator(engine,x1,x2,1000),
     a_(a),b_(b),valid(false) {

    if(a>0&&b>0) 
      {
	valid=true;
	initialize();
      }
  }

  /// Default destructor
  virtual ~GammaNumericalGenerator() {}

  /// Random generator 
  double gamma() const { return generate(); }
  
  double gamma_exp() const {return generateExp();}

  double gamma_lin() const {return generateLin();}

  /// The probability density function implementation
  virtual double function(double x) { return ersatzt(x); }

  inline bool isValid() const {return valid;}

 private:

  /// Gamma Function
  double ersatzt(double x) { 
    double bt=b_*x;
    return  b_*pow(bt,a_-1)*exp(-bt);
  }
  
  // gamma distribution parameters 
  double a_,b_;
  bool valid;
};

#endif
