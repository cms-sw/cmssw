#ifndef binomial_noncentral_intervals_h
#define binomial_noncentral_intervals_h

#include "PhysicsTools/RooStatsCms/interface/binomial_noncentral_interval.h"

struct sterne_sorter {
  bool operator()(const prob_helper& l, const prob_helper& r) const {
    return l.prob() > r.prob();
  }
};

class sterne : public binomial_noncentral_interval<sterne_sorter> {
  const char* name() const { return "Sterne"; }
};

struct feldman_cousins_sorter {
  bool operator()(const prob_helper& l, const prob_helper& r) const {
    return l.lratio() > r.lratio();
  }
};

class feldman_cousins : public binomial_noncentral_interval<feldman_cousins_sorter> {
  const char* name() const { return "Feldman-Cousins"; }
};

#endif
