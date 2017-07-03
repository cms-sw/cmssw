#ifndef CondEx_Efficiency_H
#define CondEx_Efficiency_H
/*  example of polymorphic condition
 *  LUT, function, mixed....
 * this is just a prototype: classes do not need to be defined and declared in the same file
 * at the moment though all derived classes better sit in the same package together with the base one
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include<cmath>
#include<iostream>

namespace condex {



  /* very simple base class
   * trivial inheritance, no template tricks 
   */
  class Efficiency {
  public:
    Efficiency(){}
    virtual ~Efficiency(){}
    virtual void initialize(){ std::cout << "initializing base class Efficiency" <<std::endl;}
    float operator()(float pt, float eta) const {
      return value(pt,eta);
    }

    virtual float value(float pt, float eta) const=0;

  
  COND_SERIALIZABLE;
};


  class ParametricEfficiencyInPt : public Efficiency {
  public:
    ParametricEfficiencyInPt() : cutLow(0), cutHigh(0), low(0), high(0){}
    ParametricEfficiencyInPt(float cm, float ch,
			    float el, float eh) :
      cutLow(cm), cutHigh(ch),
      low(el), high(eh){}
  private:
    float value(float pt, float) const override {
      if ( pt<low) return cutLow;
      if ( pt>high) return cutHigh;
      return cutLow + (pt-low)/(high-low)*(cutHigh-cutLow);
    }
    float cutLow, cutHigh;
    float low, high;

  
  COND_SERIALIZABLE;
};  

class ParametricEfficiencyInEta : public Efficiency {
  public:
  ParametricEfficiencyInEta() : cutLow(0), cutHigh(0), low(0), high(0) {}
    ParametricEfficiencyInEta(float cmin, float cmax,
			    float el, float eh) :
      cutLow(cmin), cutHigh(cmax),
      low(el), high(eh){}
  private:
    float value(float, float eta) const override {
      eta = std::abs(eta);
      if ( eta<low) return cutLow;
      if ( eta>high) return cutHigh;
      return cutLow + (eta-low)/(high-low)*(cutHigh-cutLow);
    }
    float cutLow, cutHigh;
    float low, high;

  
  COND_SERIALIZABLE;
};

}




#endif
