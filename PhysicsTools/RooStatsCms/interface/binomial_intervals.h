#ifndef binomial_intervals_h
#define binomial_intervals_h

#if (defined (STANDALONE) or defined (__CINT__) )
#include "binomial_interval.h"
#else
#include "PhysicsTools/RooStatsCms/interface/binomial_interval.h"
#endif

// A class to implement the calculation of intervals for the binomial
// parameter rho. The bulk of the work is done by derived classes that
// implement calculate() appropriately.

class clopper_pearson : public binomial_interval {
 public:
  void calculate(const double successes, const double trials);
  const char* name() const { return "Clopper-Pearson"; }

#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(clopper_pearson,1)
#endif
};

#endif
