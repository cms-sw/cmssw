#ifndef PhysicsTools_Utilities_Convolution_h
#define PhysicsTools_Utilities_Convolution_h
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/interface/NumericalIntegration.h"

namespace funct {
  template<typename A, typename B, typename Integrator>
  class ConvolutionStruct {
  public:
    // min and max are defined in the domain of b
    ConvolutionStruct(const A& a, const B& b, 
		      double min, double max, const Integrator & integrator) : 
      f_(a, b), min_(min), max_(max), integrator_(integrator) { 
      if(max < min)
	throw edm::Exception(edm::errors::Configuration)
	  << "Convolution: min must be smaller than max\n"; 
    }
    double operator()(double x) const {
      f_.setX(x);
      return integrator_(f_, x - max_, x - min_);
    }
   private:
    struct function {
      function(const A& a, const B& b) : _1(a), _2(b) { }
      void setX(double x) const { x_ = x; }
      double operator()(double y) const {
	return _1(y) * _2(x_ - y);
      }
    private:
      A _1;
      B _2;
      mutable double x_;
    };
    function f_;
    double min_, max_, delta_;
    Integrator integrator_;
  };

  template<typename A, typename B, typename Integrator>
  struct Convolution {
    typedef ConvolutionStruct<A, B, Integrator> type;
    static type compose(const A& a, const B& b, double min, double max, const Integrator& i) {
      return type(a, b, min, max, i);
    }
  };

  template<typename A, typename B, typename Integrator>
  inline typename funct::Convolution<A, B, Integrator>::type conv(const A& a, const B& b, double min, double max, const Integrator& i) {
    return funct::Convolution<A, B, Integrator>::compose(a, b, min, max, i);
  }

}



#endif
