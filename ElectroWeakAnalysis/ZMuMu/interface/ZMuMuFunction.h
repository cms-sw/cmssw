#ifndef ElectroWeakAnalysis_ZMuMu_ZMuMuFunction_h
#define ElectroWeakAnalysis_ZMuMu_ZMuMuFunction_h
#include "PhysicsTools/Utilities/interface/Convolution.h"
#include "PhysicsTools/Utilities/interface/ZLineShape.h"
#include "PhysicsTools/Utilities/interface/Gaussian.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuMuFunction {
   public:
    static const unsigned int arguments = 1;
    ZMuMuFunction(const Parameter & m, const Parameter & g, 
		  const Parameter & Ng, const Parameter & Ni, 
		  const Parameter & me, const Parameter & s, 
		  const Parameter & N, const Parameter & eff_tr, const Parameter & eff_sa):
      mass(m.ptr()), width(g.ptr()), 
      Ngamma(Ng.ptr()), Nint(Ni.ptr()), 
      mean(me.ptr()), sigma(s.ptr()), 
      numberOfEvents(N.ptr()), 
      efficiencyTrack(eff_tr.ptr()), efficiencyStandalone(eff_sa.ptr()), 
      zls_(m, g, Ng, Ni), gau_(me, s), 
      cgz_(zls_, gau_, -3*s.value(), 3*s.value(), 200) {}
    ZMuMuFunction(boost::shared_ptr<double> m, boost::shared_ptr<double> g, 
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
      cgz_(zls_, gau_, -3*(*s), 3*(*s), 200) {}
    double operator()(double x) const {
      double eff_tr_2 = *efficiencyTrack * (*efficiencyTrack);
      double eff_sa_2 = *efficiencyStandalone * (*efficiencyStandalone);
      return cgz_(x) * (*numberOfEvents) * eff_tr_2 * eff_sa_2;
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
