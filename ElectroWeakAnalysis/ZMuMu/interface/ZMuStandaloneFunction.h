#ifndef ElectroWeakAnalysis_ZMuMu_ZMuStandaloneFunction_h
#define ElectroWeakAnalysis_ZMuMu_ZMuStandaloneFunction_h
#include "PhysicsTools/Utilities/interface/ConvGaussZLineShape.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuStandaloneFunction {
   public:
    enum{ arguments = 1 };
    enum{ parameters = 9 };
    ZMuStandaloneFunction(boost::shared_ptr<double> m, boost::shared_ptr<double> g, boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni, 
			  boost::shared_ptr<double> me, boost::shared_ptr<double> s, 
			  boost::shared_ptr<double> N, boost::shared_ptr<double> eff_tr, boost::shared_ptr<double> eff_sa):
      mass(m), width(g), Ngamma(Ng), Nint(Ni), mean(me), sigma(s), 
      numberOfEvents(N), efficiencyTrack(eff_tr), efficiencyStandalone(eff_sa), 
      cgz_(m, g, Ng, Ni, me, s) {}
    ZMuStandaloneFunction(double m, double g, double Ng, double Ni, 
			  double me, double s, 
			  double N, double eff_tr, double eff_sa):
      mass(new double(m)), width(new double(g)), Ngamma(new double(Ng)), Nint(new double(Ni)), mean(new double(me)), sigma(new double(s)), 
      numberOfEvents(new double(N)), efficiencyTrack(new double(eff_tr)), efficiencyStandalone(new double(eff_sa)), 
      cgz_(m, g, Ng, Ni, me, s) {}
    double operator()(double x) const {
      double eff_sa_2 = *efficiencyStandalone * (*efficiencyStandalone);
      double eff_tr_minus = *efficiencyTrack * ( 1. - *efficiencyTrack );
      return cgz_(x) * 2. * (*numberOfEvents) * eff_sa_2 * eff_tr_minus;
    } 
    void setParameters(double m, double g, double Ng, double Ni, 
		       double me, double s, 
		       double N, double eff_tr, double eff_sa) {
      *mass = m; 
      *width = g; 
      *Ngamma = Ng; 
      *Nint = Ni; 
      *mean = me; 
      *sigma = s; 
      *numberOfEvents = N; 
      *efficiencyTrack = eff_tr; 
      *efficiencyStandalone = eff_sa; 
      cgz_.setParameters(m, g, Ng, Ni, me, s);
    }
    boost::shared_ptr<double> mass, width, Ngamma, Nint, mean, sigma;
    boost::shared_ptr<double> numberOfEvents, efficiencyTrack, efficiencyStandalone;
  private:
    ConvGaussZLineShape cgz_;
  };
}

#endif
