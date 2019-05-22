#ifndef _COMMONDET_MEASUREMENT1DFLOAT_H_
#define _COMMONDET_MEASUREMENT1DFLOAT_H_

/** A class that combines a value and it's associated uncertainty,
 *  or error, together. Provides a more explicit interface than
 *  a pair<float,float>. If you don't like the name, propose a better one!
 */

class Measurement1DFloat {
public:
  // construct

  Measurement1DFloat() : theValue(0.), theError(0.) {}

  Measurement1DFloat(const float& aValue) : theValue(aValue), theError(0.) {}

  Measurement1DFloat(const float& aValue, const float& aError) : theValue(aValue), theError(aError) {}

  //destruct

  ~Measurement1DFloat(){};

  float value() const { return theValue; }

  float error() const { return theError; }

  float significance() const {
    if (theError == 0)
      return 0;
    else
      return theValue / theError;
  }

private:
  float theValue;
  float theError;
};

#endif
