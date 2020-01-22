#ifndef UTIL_H
#define UTIL_H

class Util {

public:
  
  //method return phi in the -pi to +pi range
  static double phiRange(double phi){
    //catch if phi is very out of range, not a number etc
    assert(fabs(phi)<100.0);
    while(phi<-M_PI) phi+=2*M_PI;
    while(phi>M_PI) phi-=2*M_PI;
    return phi;
  }


};

#endif
