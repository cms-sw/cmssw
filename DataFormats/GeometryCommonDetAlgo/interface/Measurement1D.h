#ifndef _COMMONDET_MEASUREMENT1D_H_
#define _COMMONDET_MEASUREMENT1D_H_

#include <string>

/** A class that combines a value and it's associated uncertainty,
 *  or error, together. Provides a more explicit interface than
 *  a pair<double,double>. If you don't like the name, propose a better one!
 */

class Measurement1D {

public:
// construct

Measurement1D() : theValue(0.) , theError(0.) {};

Measurement1D( const double& aValue) : 
  theValue(aValue) , theError(0.) {};

Measurement1D( const double& aValue, const double& aError) 
  : theValue(aValue) , theError(aError) {}; 

//destruct

~Measurement1D() {} ;

double value() const { return theValue;}

double error() const { return theError;}

double significance() const {
  if (theError == 0) return 0;
  else return theValue/theError;
}

private:

double  theValue;
double  theError;

};



#endif










