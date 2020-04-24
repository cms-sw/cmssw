#ifndef PhysicsTools_Utilities_Likelihood_h
#define PhysicsTools_Utilities_Likelihood_h
#include "PhysicsTools/Utilities/interface/RootMinuitResultPrinter.h"
#include "PhysicsTools/Utilities/interface/RootMinuitFuncEvaluator.h"
#include <cmath>
#include "TMath.h"

namespace fit {
  template<typename PDF, typename Tuple>
  struct LikelihoodEvaluator {
    static double evaluate(const PDF & pdf, const Tuple & tuple) {
      return pdf(tuple);
    }
  };

  // to be generalized for tuples in N variables
  template<typename PDF>
  struct LikelihoodEvaluator<PDF, double> {
    static double evaluate(const PDF & pdf, const double val) {
      return pdf(val);
    }    
  };

  struct NoExtendedLikelihood { };

  template<typename Sample, typename PDF, typename Yield = NoExtendedLikelihood>
  class Likelihood {
  public:
    Likelihood() { }
    Likelihood(const Sample & sample, PDF & pdf, Yield & yield) : 
      pdf_(& pdf), yield_(&yield), sample_(sample) { 
    }
    double operator()() const { return log(); }
    double log() const {
      double l = - (*yield_)();
      for(typename Sample::const_iterator i = sample_.begin(); i != sample_.end(); ++i) {
	double p = Evaluator::evaluate(*pdf_, *i);
	l += std::log(p);
      }
      sampleSize_ = sample_.size();
      return l;
    }
    double logNFactorial() const {
      return std::log(TMath::Factorial(sampleSize_));
    }
    double absoluteLog() const {
      return log() - logNFactorial();
    }
    PDF & pdf() { return * pdf_; }
    const PDF & pdf() const { return * pdf_; }
    Yield & yield() { return * yield_; }
    const Yield & yield() const { return * yield_; }
    unsigned int sampleSize() const { return sampleSize_; }
  private:
    typedef LikelihoodEvaluator<PDF, typename Sample::value_type> Evaluator;
    PDF * pdf_;
    Yield * yield_;
    Sample sample_;
    mutable unsigned int sampleSize_ = 0u;
  };

  template<typename Sample, typename PDF>
  class Likelihood<Sample, PDF, NoExtendedLikelihood> {
  public:
    Likelihood() { }
    Likelihood(const Sample & sample, PDF & pdf) : 
      pdf_(& pdf), sample_(sample) { 
    }
    double operator()() const { return log(); }
    double log() const {
      double l = 0;
      for(typename Sample::const_iterator i = sample_.begin(); i != sample_.end(); ++i) {
	l += std::log(Evaluator::evaluate(*pdf_, *i));
      }
      return l;
    }
    PDF & pdf() { return * pdf_; }
    const PDF & pdf() const { return * pdf_; }
  private:
    typedef LikelihoodEvaluator<PDF, typename Sample::value_type> Evaluator;
    PDF * pdf_;
    Sample sample_;
  };

  template<typename Sample, typename PDF, typename Yield>
  struct RootMinuitResultPrinter<Likelihood<Sample, PDF, Yield> > {
    static void print(double amin, unsigned int numberOfFreeParameters, const Likelihood<Sample, PDF, Yield> & f) {
      std::cout << "-2 log(maximum-likelihood) = " << amin << ", free parameters = " << numberOfFreeParameters
		<< std::endl;      
    }
  };

  template<typename Sample, typename PDF, typename Yield>
  struct RootMinuitFuncEvaluator<Likelihood<Sample, PDF, Yield> > {
    static double evaluate(const Likelihood<Sample, PDF, Yield> & f) {
      return - 2 * f();
    }
  };

}

#endif
