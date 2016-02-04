#ifndef ElectroWeakAnalysis_ZMuMu_ZMuStandaloneFunction_h
#define ElectroWeakAnalysis_ZMuMu_ZMuStandaloneFunction_h
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuStandaloneFunction {
   public:
    static const unsigned int arguments = 1;
    ZMuStandaloneFunction(boost::shared_ptr<double> m, boost::shared_ptr<double> g, 
			  boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni, 
			  boost::shared_ptr<double> me, boost::shared_ptr<double> s, 
			  boost::shared_ptr<double> N, 
			  boost::shared_ptr<double> eff_tr, boost::shared_ptr<double> eff_sa):
      mass(m), width(g), 
      Ngamma(Ng), Nint(Ni), 
      mean(me), sigma(s), 
      numberOfEvents(N), 
      efficiencyTrack(eff_tr), efficiencyStandalone(eff_sa), 
      zls_(m, g, Ng, Ni), gau_(me, s), 
      cgz_(zls_, gau_, -3*s.value(), 3*s.value(), 200) {}
    ZMuStandaloneFunction(double m, double g, double Ng, double Ni, 
			  double me, double s, 
			  double N, double eff_tr, double eff_sa):
      mass(new double(m)), width(new double(g)), 
      Ngamma(new double(Ng)), Nint(new double(Ni)), 
      mean(new double(me)), sigma(new double(s)), 
      numberOfEvents(new double(N)), 
      efficiencyTrack(new double(eff_tr)), efficiencyStandalone(new double(eff_sa)), 
      zls_(m, g, Ng, Ni), gau_(me, s), 
      cgz_(zls_, gau_, -3*(*s), 3*(*s), 200) {}
    double operator()(double x) const {
      double eff_sa_2 = *efficiencyStandalone * (*efficiencyStandalone);
      double eff_tr_minus = *efficiencyTrack * ( 1. - *efficiencyTrack );
      return cgz_(x) * 2. * (*numberOfEvents) * eff_sa_2 * eff_tr_minus;
    } 
    boost::shared_ptr<double> mass, width, Ngamma, Nint, mean, sigma;
    boost::shared_ptr<double> numberOfEvents, efficiencyTrack, efficiencyStandalone;
  private:
    ZLineShape zls_;
    Gaussian gau_;
    Convolution<ZLineShape, Gaussian> cgz_;
  };
}

#endif
