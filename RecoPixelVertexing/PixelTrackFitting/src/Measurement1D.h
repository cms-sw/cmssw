#ifndef Measurement1D_H
#define Measurement1D_H

#include <utility>
#include <iostream>
#include <algorithm>

using namespace std;

class Measurement1D : public pair<float,float> {
public:
  Measurement1D() 
      : pair<float,float>(0.,0.) { }
  Measurement1D(const double& aValue, const double& aError) 
      : pair<float,float>(aValue,aError) { }

  float value() const { return this->first;}
  float error() const { return this->second;}

};

#endif
