#ifndef PhysicsTools_Utilities_Convolution_h
#define PhysicsTools_Utilities_Convolution_h
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/interface/NumericalIntegration.h"

namespace funct {
  template<typename A, typename B>
  class ConvolutionStruct {
   public:
    // min and max are defined in the domain of b
    ConvolutionStruct(const A& a, const B& b, 
		      double min, double max, size_t samples) : 
      f_(a, b), min_(min), max_(max), delta_((max-min)/(samples-1)), samples_(samples) { 
      if(max < min)
	throw edm::Exception(edm::errors::Configuration)
	  << "Convolution: min must be smaller than max\n"; 
    }
    double operator()(double x) const {
      f_.setX(x);
      return trapezoid_integral(f_, min_, max_, samples_);
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
    size_t samples_;
  };

  /*
  template<typename A, typename B>
  double ConvolutionStruct<A, B>::operator()(double x) const {
    double f = 0; 
    // x - max < y < x - min
    double y0 = x - max_;
    for(size_t n = 0; n < samples_; ++n) {
      double y = y0 + n*delta_;
      f += _1(y) * _2(x - y);
    }   
    return f * delta_;
  }
  */

  template<typename A, typename B>
  struct Convolution {
    typedef ConvolutionStruct<A, B> type;
    static type compose(const A& a, const B& b, double min, double max, size_t samples) {
      return type(a, b, min, max, samples);
    }
  };

  template<typename A, typename B>
  inline typename funct::Convolution<A, B>::type conv(const A& a, const B& b, double min, double max, size_t samples) {
    return funct::Convolution<A, B>::compose(a, b, min, max, samples);
  }

}



#endif
