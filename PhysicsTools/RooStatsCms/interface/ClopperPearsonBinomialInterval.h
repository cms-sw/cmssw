#ifndef PhysicsTools_RooStatsCms_ClopperPearsonBinomialInterval_h
#define PhysicsTools_RooStatsCms_ClopperPearsonBinomialInterval_h
/* \class ClopperPearsonBinomialInterval
 *
 * \author Jordan Tucker
 *
 * integration in CMSSW: Luca Lista
 *
 */

#if (defined (STANDALONE) or defined (__CINT__) )
#include "BinomialInterval.h"
#else
#include "PhysicsTools/RooStatsCms/interface/BinomialInterval.h"
#endif

// A class to implement the calculation of intervals for the binomial
// parameter rho. The bulk of the work is done by derived classes that
// implement calculate() appropriately.

class ClopperPearsonBinomialInterval : public BinomialInterval {
 public:
  void calculate(const double successes, const double trials);
  const char* name() const { return "Clopper-Pearson"; }

#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(ClopperPearsonBinomialInterval,1)
#endif
};

#endif
