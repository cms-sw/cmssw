#ifndef PhysicsTools_Utilities_CombinedChiSquaredLikelihood_h
#define PhysicsTools_Utilities_CombinedChiSquaredLikelihood_h
#include "PhysicsTools/Utilities/interface/RootMinuitResultPrinter.h"
#include "PhysicsTools/Utilities/interface/RootMinuitFuncEvaluator.h"

namespace fit {
  template<typename ChiSquared, typename Likelihood>
  class CombinedChiSquaredLikelihood {
  public:
    CombinedChiSquaredLikelihood() { }
    CombinedChiSquaredLikelihood(const ChiSquared & chi2, const Likelihood & like) :
      chi2_(chi2), like_(like) { }
    // return chi-square value
    double operator()() const { 
      return - 2 * like_() + chi2_();
    }
    ChiSquared & chi2() { return chi2_; }
    const ChiSquared & chi2() const { return chi2_; }
    Likelihood & like() { return like_; }
    const Likelihood & like() const { return like_; }
  private:
    ChiSquared chi2_;
    Likelihood like_;
  };

  template<typename ChiSquared, typename Likelihood>
  struct RootMinuitResultPrinter<CombinedChiSquaredLikelihood<ChiSquared, Likelihood> > {
    static void print(double amin, unsigned int numberOfFreeParameters, const CombinedChiSquaredLikelihood<ChiSquared, Likelihood> & f) {
      std::cout << "-2 log(maximum-likelihood) = " << amin << ", free parameters = " << numberOfFreeParameters
		<< std::endl;      
      std::cout << "chi-2 contibution: " << f.chi2()() << std::endl
		<< "likelihood contriution: " << -2.*f.like()() << std::endl;
    }
  };

}

#endif
