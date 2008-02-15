#ifndef ElectroWeakAnalysis_ZMuMu_ZMuMuBack_h
#define ElectroWeakAnalysis_ZMuMu_ZMuMuBack_h

#include "PhysicsTools/Utilities/interface/Exponential.h"
#include "PhysicsTools/Utilities/interface/Polynomial.h"
#include <boost/shared_ptr.hpp>

namespace function{
  class ZMuMuBack {
   public:
    enum{ arguments = 1 };
    enum{ parameters = 4 };
    ZMuMuBack(boost::shared_ptr<double> Nb, boost::shared_ptr<double> l, 
	      boost::shared_ptr<double> a, boost::shared_ptr<double> b):
      Nbkg(Nb), lambda(l), a1(a), a2(b), 
      expo_(-(*l)), poly_(1., *a, *b) {}
    ZMuMuBack(double Nb, double l, double a, double b):
      Nbkg(new double(Nb)), lambda(new double(l)), a1(new double(a)), a2(new double(b)), 
      expo_(-l), poly_(1, a, b) {}
    double operator()(double x) const  {
      return *Nbkg * expo_(x) * poly_(x);
    }
    void setParameters(double Nb, double l, double a, double b) {
      *Nbkg = Nb; 
      *lambda = l;
      *a1 = a;
      *a2 = b;
      expo_.setParameters(-l);
      poly_.setParameters(1., a, b);
    }
    
    boost::shared_ptr<double> Nbkg, lambda, a1, a2;
  private:
    Exponential expo_;
    Polynomial<2> poly_;
  };
}

#endif
