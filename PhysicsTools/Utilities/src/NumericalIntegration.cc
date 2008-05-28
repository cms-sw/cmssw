#include "PhysicsTools/Utilities/interface/NumericalIntegration.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <cmath>

funct::GaussLegendreIntegrator::GaussLegendreIntegrator(unsigned int samples, double epsilon) :
  samples_(samples) {
  if (samples <= 0)
    throw edm::Exception(edm::errors::Configuration)
      << "gauss_legendre_integral: number of samples must be positive\n"; 
  if(epsilon <= 0)
    throw edm::Exception(edm::errors::Configuration)
      << "gauss_legendre_integral: numerical precision must be positive\n"; 
  
  x.resize(samples);
  w.resize(samples);
  const unsigned int m = (samples + 1)/2;
  
  double z, zSqr, pp, p1, p2, p3;
  
  for (unsigned int i = 0; i < m; ++i) {
    z = std::cos(3.14159265358979323846 * (i + 0.75)/(samples + 0.5));
    zSqr = z*z;
    do {
      p1 = 1.0;
      p2 = 0.0;
      for (unsigned int j = 0; j < samples; ++j) {
	p3 = p2;
	p2 = p1;
	p1 = ((2.0*j + 1.0)*z*p2 - j*p3)/(j + 1.0);
      }
      pp = samples*(z*p1 - p2)/(zSqr - 1.0);
      z -= p1/pp;
    } while (std::fabs(p1/pp) > epsilon);
    
    x[i] = -z;
    x[samples - i - 1] = z;
    w[i] = 2.0/((1.0 - zSqr)*pp*pp);
    w[samples - i -1] = w[i];
  }
}
