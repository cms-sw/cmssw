#ifndef ElectroWeakAnalysis_ZMuMu_ZMuStandaloneScaledFunction_h
#define ElectroWeakAnalysis_ZMuMu_ZMuStandaloneScaledFunction_h
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuStandaloneFunction.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuStandaloneScaledFunction {
   public: 
    enum { arguments = 1 };
    enum { parameters = 9 };
    ZMuStandaloneScaledFunction(const ZMuStandaloneFunction & zms, int bin): 
      binScaleFactor(bin), zms_(zms) {}
    ZMuStandaloneScaledFunction(boost::shared_ptr<double> m, boost::shared_ptr<double> g, boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni, 
				boost::shared_ptr<double> me, boost::shared_ptr<double> s, 
				boost::shared_ptr<double> N, boost::shared_ptr<double> eff_tr, boost::shared_ptr<double> eff_sa, int bin):
      mass(m), width(g), Ngamma(Ng), Nint(Ni), mean(me), sigma(s), 
      numberOfEvents(N), efficiencyTrack(eff_tr), efficiencyStandalone(eff_sa), binScaleFactor(bin), 
      zms_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa) {}
    ZMuStandaloneScaledFunction(double m, double g, double Ng, double Ni, 
				double me, double s, 
				double N, double eff_tr, double eff_sa, 
				int bin): 
      mass(new double(m)), width(new double(g)), Ngamma(new double(Ng)), Nint(new double(Ni)), mean(new double(me)), sigma(new double(s)), 
      numberOfEvents(new double(N)), efficiencyTrack(new double(eff_tr)), efficiencyStandalone(new double(eff_sa)), binScaleFactor(bin), 
      zms_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa) {}
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
      zms_.setParameters(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa);
    }
    double operator()(double x) const {
      return binScaleFactor * zms_(x);
    }
    boost::shared_ptr<double> mass, width, Ngamma, Nint, mean, sigma;
    boost::shared_ptr<double> numberOfEvents, efficiencyTrack, efficiencyStandalone;
    int binScaleFactor;
  private:
    ZMuStandaloneFunction zms_;
  };
}  

#endif
  
