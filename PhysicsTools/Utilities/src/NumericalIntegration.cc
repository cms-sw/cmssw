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


const double funct::GaussIntegrator::x[12] = { 
  0.96028985649753623, 0.79666647741362674,
  0.52553240991632899, 0.18343464249564980,
  0.98940093499164993, 0.94457502307323258,
  0.86563120238783174, 0.75540440835500303,
  0.61787624440264375, 0.45801677765722739,
  0.28160355077925891, 0.09501250983763744 };
     
const double funct::GaussIntegrator::w[12] = { 
  0.10122853629037626, 0.22238103445337447,
  0.31370664587788729, 0.36268378337836198,
  0.02715245941175409, 0.06225352393864789,
  0.09515851168249278, 0.12462897125553387,
  0.14959598881657673, 0.16915651939500254,
  0.18260341504492359, 0.18945061045506850 };

const double funct::GaussIntegrator::kHF = 0.5;
const double funct::GaussIntegrator::kCST = 5./1000;
