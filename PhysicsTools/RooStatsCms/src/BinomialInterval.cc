#include <string>
#include <cstdio>
#include "Math/PdfFuncMathCore.h"
#include "Math/QuantFuncMathCore.h"

#if (defined (STANDALONE) or defined (__CINT__) )
#include "BinomialInterval.h"

ClassImp(BinomialInterval)
#else
#include "PhysicsTools/RooStatsCms/interface/BinomialInterval.h"
#endif

void BinomialInterval::init(const double alpha, const tail_type type) {
  alpha_     = alpha;
  type_      = type;
  alpha_min_ = type_ == equal_tailed ? alpha_/2 : alpha_;
  kappa_     = ROOT::Math::normal_quantile(1 - alpha/2, 1);
  kappa2_    = kappa_*kappa_;
}

bool BinomialInterval::contains(double p) {
  if (type_ == upper_tailed)
    return p <= upper_;
  else if (type_ == lower_tailed)
    return p >= lower_;
  else //if (type_ == equal_tailed)
    return p >= lower_ && p <= upper_;
}

double BinomialInterval::coverage_prob(const double p, const int trials) {
  double prob = 0;

  for (int X = 0; X <= trials; ++X) {
    calculate(X, trials);

    if (contains(p))
      prob += ROOT::Math::binomial_pdf(X, p, trials);
  }
  
  return prob;
}

void BinomialInterval::scan_rho(const int ntot, const int nrho, double* rho, double* prob) {
  for (int i = 0; i < nrho; ++i) {
    rho[i]  = double(i)/nrho;
    prob[i] = coverage_prob(rho[i], ntot);
  }
}

void BinomialInterval::scan_ntot(const double rho, const int ntot_min, const int ntot_max,
				  double* ntot, double* prob) {
  for (int i = 0; i < ntot_max - ntot_min + 1; ++i) {
    int nt = i + ntot_min;
    ntot[i] = nt;
    prob[i] = coverage_prob(rho, nt);
  }
}

void BinomialInterval::dump(const int trials_min, const int trials_max) {
  const std::string fn = std::string("table.") + name() + std::string(".txt");
  FILE* fdump = fopen(fn.c_str(), "wt");

  for (int n = trials_min; n <= trials_max; ++n) {
    for (int X = 0; X <= n; X++) {
      calculate(X, n);
      fprintf(fdump, "%i %i %f %f\n", X, n, lower_, upper_);
    }
    fprintf(fdump, "\n");
  }

  fclose(fdump);
}
