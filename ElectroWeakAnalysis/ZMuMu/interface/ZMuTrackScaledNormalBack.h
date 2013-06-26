#ifndef ElectroWeakAnalysis_ZMuMu_ZMuTrackScaledNormalBack_h
#define ElectroWeakAnalysis_ZMuMu_ZMuTrackScaledNormalBack_h
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuTrackScaledFunction.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuNormalBack.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuTrackScaledNormalBack { 
   public:
    enum{ arguments = 1}; 
    enum{ parameters = 13};
    ZMuTrackScaledNormalBack(boost::shared_ptr<double> m, boost::shared_ptr<double> g, boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni, 
			     boost::shared_ptr<double> me, boost::shared_ptr<double> s, 
			     boost::shared_ptr<double> N, boost::shared_ptr<double> eff_tr, boost::shared_ptr<double> eff_sa, 
			     boost::shared_ptr<double> Nb, boost::shared_ptr<double> l, boost::shared_ptr<double> a, boost::shared_ptr<double> b, 
			     int bin, int rmin, int rmax):
      mass(m), width(g), Ngamma(Ng), Nint(Ni), 
      mean(me), sigma(s), 
      numberOfEvents(N), efficiencyTrack(eff_tr), efficiencyStandalone(eff_sa), 
      Nbkg(Nb), lambda(l), a1(a), a2(b), 
      binScaleFactor(bin), x_min(rmin), x_max(rmax), 
      zmts_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa, bin), zmmnb_(Nb, l, a, b, rmin, rmax) {}
    ZMuTrackScaledNormalBack(double m, double g, double Ng, double Ni, 
			     double me, double s, 
			     double N, double eff_tr, double eff_sa, 
			     double Nb, double l, double a, double b, 
			     int bin, int rmin, int rmax): 
      mass(new double(m)), width(new double(g)), Ngamma(new double(Ng)), Nint(new double(Ni)), 
      mean(new double(me)), sigma(new double(s)), 
      numberOfEvents(new double(N)), efficiencyTrack(new double(eff_tr)), efficiencyStandalone(new double(eff_sa)), 
      Nbkg(new double(Nb)), lambda(new double(l)), a1(new double(a)), a2(new double(b)), 
      binScaleFactor(bin), x_min(rmin), x_max(rmax), 
      zmts_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa, bin), zmmnb_(Nb, l, a, b, rmin, rmax) {}
    void setParameters(double m, double g, double Ng, double Ni, 
		       double me, double s, 
		       double N, double eff_tr, double eff_sa, 
		       double Nb, double l, double a, double b) { 
      zmts_.setParameters(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa);
      zmmnb_.setParameters(Nb, l, a, b);
    } 
    double operator()(double x) const { 
      return zmts_(x) + zmmnb_(x);
    }
    boost::shared_ptr<double> mass, width, Ngamma, Nint, mean, sigma;
    boost::shared_ptr<double> numberOfEvents, efficiencyTrack, efficiencyStandalone;
    boost::shared_ptr<double> Nbkg, lambda, a1, a2;
    int binScaleFactor, x_min, x_max; 
  private:
    ZMuTrackScaledFunction zmts_; 
    ZMuMuNormalBack zmmnb_;
  };
}

#endif
