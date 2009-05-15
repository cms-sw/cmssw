#ifndef binomial_intervals_h
#define binomial_intervals_h

#include "binomial_interval.h"

class clopper_pearson : public binomial_interval {
 public:
  void calculate(const double successes, const double trials);
  const char* name() const { return "Clopper-Pearson"; }
};

#endif
