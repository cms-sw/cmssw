#ifndef PhysicsTools_Utilities_NumericalIntegration_h
#define PhysicsTools_Utilities_NumericalIntegration_h

namespace funct {
  
  template<typename F>
  double trapezoid_integral(const F& f, double min, double max, unsigned int samples) { 
    const double l = max - min, delta = l / samples;
    double sum = 0;
    for(unsigned int i = 0; i < samples; ++i) 
      sum += f(min + (i + 0.5) * delta);
    return sum * delta;
  }
  
}

#endif
