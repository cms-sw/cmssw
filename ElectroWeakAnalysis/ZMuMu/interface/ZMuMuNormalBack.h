#ifndef ElectroWeakAnalysis_ZMuMu_ZMuMuNormalBack_h
#define ElectroWeakAnalysis_ZMuMu_ZMuMuNormalBack_h
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuBack.h"
#include "ElectroWeakAnalysis/ZMuMu/interface/ZMuMuBackNorm.h"
#include <boost/shared_ptr.hpp>

namespace function {
  class ZMuMuNormalBack { 
   public:
    enum{ arguments = 1}; 
    enum{ parameters = 4};
    ZMuMuNormalBack(boost::shared_ptr<double> Nb, boost::shared_ptr<double> l, boost::shared_ptr<double> a, boost::shared_ptr<double> b, 
		    int rmin, int rmax): 
      Nbkg(Nb), lambda(l), a1(a), a2(b), x_min(rmin), x_max(rmax), 
      zmb_(Nb, l, a, b), zmbn_(l, a, b) {}
    ZMuMuNormalBack(double Nb, double l, double a, double b, 
		    int rmin, int rmax):
      Nbkg(new double(Nb)), lambda(new double(l)), a1(new double(a)), a2(new double(b)), x_min(rmin), x_max(rmax), 
      zmb_(Nb, l, a, b), zmbn_(l, a, b) {} 
    void setParameters(double Nb, double l, double a, double b) { 
      zmb_.setParameters(Nb, l, a, b);
      zmbn_.setParameters(l, a, b);
    } 
    double operator()(double x) const { 
      return zmbn_(x_min, x_max) * zmb_(x);
    }
    boost::shared_ptr<double> Nbkg, lambda, a1, a2;
    int x_min, x_max;
  private:
    ZMuMuBack zmb_;
    ZMuMuBackNorm zmbn_;
  };
}

#endif
