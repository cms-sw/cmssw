#include "PhysicsTools/RooStatsCms/interface/binomial_noncentral_intervals.h"

int main() {
  feldman_cousins fc;
  //  const double alpha = 0.05; // 1 - CL
  const double alpha = (1-0.682);
  fc.init(alpha);

  const int ntrials = 50;
  for (int nsuccesses = 0; nsuccesses <= ntrials; ++nsuccesses) {
    fc.calculate(nsuccesses, ntrials);
    double eff = double(nsuccesses)/double(ntrials);
    double err_minus = eff - fc.lower(), err_plus = fc.upper() - eff;
    double err = sqrt(eff*(1.0-eff)/ntrials);
    printf("nsuccesses: %3i  ntrials: %3i  Feldman-Cousins lower endpoint: %f upper endpoint: %f\n eff = %f +/- [%f,%f] (%f)\n",
	   nsuccesses, ntrials, fc.lower(), fc.upper(), eff, err_minus, err_plus, err);
  }
}

