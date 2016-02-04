#ifndef ElectroWeakAnalysis_ZMuMu_ZMuMuScaledFunction_h
#define ElectroWeakAnalysis_ZMuMu_ZMuMuScaledFunction_h
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuFunction.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuMuScaledFunction {
   public: 
    static const unsigned int arguments = 1;
    ZMuMuScaledFunction(const ZMuMuFunction & zmm, int bin): 
      binScaleFactor(bin), zmm_(zmm) {}
     ZMuMuScaledFunction(const Parameter & m, const Parameter & g, 
			 const Parameter & Ng, const Parameter & Ni, 
			 const Parameter & me, const Parameter & s, 
			 const Parameter & N, 
			 const Parameter & eff_tr, const Parameter & eff_sa, 
			 int bin):
       mass(m.ptr()), width(g.ptr()), 
       Ngamma(Ng.ptr()), Nint(Ni.ptr()), 
       mean(me.ptr()), sigma(s.ptr()), 
       numberOfEvents(N.ptr()), 
       efficiencyTrack(eff_tr.ptr()), efficiencyStandalone(eff_sa.ptr()), 
       binScaleFactor(bin), 
       zmm_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa) {}
    ZMuMuScaledFunction(boost::shared_ptr<double> m, boost::shared_ptr<double> g, 
			boost::shared_ptr<double> Ng, boost::shared_ptr<double> Ni, 
			boost::shared_ptr<double> me, boost::shared_ptr<double> s, 
			boost::shared_ptr<double> N, 
			boost::shared_ptr<double> eff_tr, boost::shared_ptr<double> eff_sa, 
			int bin):
      mass(m), width(g), 
      Ngamma(Ng), Nint(Ni), 
      mean(me), sigma(s), 
      numberOfEvents(N), 
      efficiencyTrack(eff_tr), efficiencyStandalone(eff_sa), 
      binScaleFactor(bin), 
      zmm_(m, g, Ng, Ni, me, s, N, eff_tr, eff_sa) {}
    double operator()(double x) const {
      return binScaleFactor * zmm_(x);
    }
    boost::shared_ptr<double> mass, width, Ngamma, Nint, mean, sigma;
    boost::shared_ptr<double> numberOfEvents, efficiencyTrack, efficiencyStandalone;
    int binScaleFactor;
  private:
    ZMuMuFunction zmm_;
  };
}

#endif

/*template<typename F, int arguments = F::arguments, int parameters = F::parameters>
class ScaleFactor{ 
  ScaleFactor(const int);
  void setParameters(const double *);
  double operator()(const double *) const;
 private:
  int bin_;
  F f_;
};

template <typename F>
double scaleFunction(const double * x, const double * par, const int bin) {
  static ScaleFunction<F> f(bin);
  f.setParameters(par);
  return f(x);
}

template<typename F, int arguments, int parameters>
ScaleFunction<F, arguments, parameters>::ScaleFunction(const double bin) :
  bin_(bin) {
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 1>::setParameters(const double * par) {
  f_.setParameters(par[0]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 2>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 3>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 4>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 5>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 6>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4], par[5]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 7>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4], par[5], par[6]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 8>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 9>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8]);
}

template<typename F, int arguments>
void ScaleFunction<F, arguments, 10>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], par[9]);
}

template<typename F, int arguments>
void ScaleFactor<F, arguments, 11>::setParameters(const double * par) {
  f_.setParameters(par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], par[9], par[10]);
}

template<typename F, int parameters>
double ScaleFactor<F, 1, parameters>::operator()(const double * x) const{
  return f_(x[0]);
}

template<typename F, int parameters>
double ScaleFactor<F, 2, parameters>::operator()(const double * x) const{
  return f_(x[0], x[1]);
}
*/
