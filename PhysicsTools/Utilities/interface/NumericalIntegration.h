#ifndef PhysicsTools_Utilities_NumericalIntegration_h
#define PhysicsTools_Utilities_NumericalIntegration_h
/* 
 * Numerical integration utilities
 *
 * \author: Luca Lista
 * Gauss Legendre algorithm based on ROOT implementation
 *
 */
#include <vector>

namespace funct {

  template<typename F>
  double trapezoid_integral(const F& f, double min, double max, unsigned int samples) { 
      const double l = max - min, delta = l / samples;
      double sum = 0;
      for(unsigned int i = 0; i < samples; ++i) 
	sum += f(min + (i + 0.5) * delta);
      return sum * delta;
    }
  
  class TrapezoidIntegrator {
  public:
    TrapezoidIntegrator() : samples_(0) { }
    explicit TrapezoidIntegrator(unsigned int samples) : samples_(samples) { }
    template<typename F>
    double operator()(const F& f, double min, double max) const { 
      return trapezoid_integral(f, min, max, samples_);
    }
  private:
    const unsigned int samples_;
  };

  class GaussLegendreIntegrator {
  public:
    GaussLegendreIntegrator() : samples_(0) { }
    GaussLegendreIntegrator(unsigned int samples, double epsilon);
    template<typename F>
      double operator()(const F& f, double min, double max) const {
      const double a0 = 0.5*(max + min);
      const double b0 = 0.5*(max - min);
      
      double result = 0.0;
      for (unsigned int i = 0; i < samples_; ++i) {
	result += w[i] * f(a0 + b0*x[i]);
      }
      
      return result * b0;      
    }
  private:
    const unsigned int samples_;
    std::vector<double> x, w;
  };
  
}

#endif
