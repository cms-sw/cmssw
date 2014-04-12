#ifndef PhysicsTools_RooStatsCms_BinomialProbHelper_h
#define PhysicsTools_RooStatsCms_BinomialProbHelper_h
/* \class BinomialProbHelper
 * 
 * \author Jordan Tucker
 * integration in CMSSW: Luca Lista
 *
 */

class BinomialProbHelper {
public:
  BinomialProbHelper(double rho, int x, int n)
    : rho_(rho), x_(x), n_(n),
    rho_hat_(double(x)/n),
    prob_(ROOT::Math::binomial_pdf(x, rho, n)) {
    // Cache the likelihood ratio L(\rho)/L(\hat{\rho}), too.
    if (x == 0)
      lratio_ = pow(1 - rho, n);
    else if (x == n)
      lratio_ = pow(rho, n);
    else
      lratio_ = pow(rho/rho_hat_, x) * pow((1 - rho)/(1 - rho_hat_), n - x);
  }

  double rho   () const { return rho_;    };
  int    x     () const { return x_;      };
  int    n     () const { return n_;      };
  double prob  () const { return prob_;   };
  double lratio() const { return lratio_; };

 private:
  double rho_;
  int x_;
  int n_;
  double rho_hat_;
  double prob_;
  double lratio_;
};

#endif
