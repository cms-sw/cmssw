#ifndef PhysicsTools_RooStatsCms_BinomialNoncentralinterval_h
#define PhysicsTools_RooStatsCms_BinomialNoncentralinterval_h
/* \class BinomialNoncentralinterval
 *
 * \author Jordan Tucker
 *
 * integration in CMSSW: Luca Listta
 *
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "Math/PdfFuncMathCore.h"

#if (defined (STANDALONE) or defined (__CINT__) )
#include "BinomialInterval.h"
#include "BinomialProbHelper.h"
#else
#include "PhysicsTools/RooStatsCms/interface/BinomialInterval.h"
#include "PhysicsTools/RooStatsCms/interface/BinomialProbHelper.h"
#endif

// Implement noncentral binomial confidence intervals using the Neyman
// construction. The Sorter class gives the ordering of points,
// i.e. it must be a functor implementing a greater-than relationship
// between two prob_helper instances. See feldman_cousins for an
// example.
template <typename Sorter>
class BinomialNoncentralInterval : public BinomialInterval {
 public:
  // Given a true value of rho and ntot trials, calculate the
  // acceptance set [x_l, x_r] for use in a Neyman construction.
  bool find_rho_set(const double rho, const int ntot, int& x_l, int& x_r) const {
    // Get the binomial probabilities for every x = 0..n, and sort them
    // in decreasing order, determined by the Sorter class.
    std::vector<BinomialProbHelper> probs;
    for (int i = 0; i <= ntot; ++i)
      probs.push_back(BinomialProbHelper(rho, i, ntot));
    std::sort(probs.begin(), probs.end(), sorter_);

    // Add up the probabilities until the total is 1 - alpha or
    // bigger, adding the biggest point first, then the next biggest,
    // etc. "Biggest" is given by the Sorter class and is taken care
    // of by the sort above. JMTBAD need to find equal probs and use
    // the sorter to differentiate between them.
    const double target = 1 - alpha_;
    // An invalid interval.
    x_l = ntot;
    x_r = 0;
    double sum = 0;
    for (int i = 0; i <= ntot && sum < target; ++i) {
      sum += probs[i].prob();
      const int& x = probs[i].x();
      if (x < x_l) x_l = x;
      if (x > x_r) x_r = x;
    }
  
    return x_l <= x_r;
  }

  // Construct nrho acceptance sets in rho = [0,1] given ntot trials
  // and put the results in already-allocated x_l and x_r.
  bool neyman(const int ntot, const int nrho, double* rho, double* x_l, double* x_r) {
    int xL, xR;
    for (int i = 0; i < nrho; ++i) {
      rho[i] = double(i)/nrho;
      find_rho_set(rho[i], ntot, xL, xR);
      x_l[i] = xL;
      x_r[i] = xR;
    }
    return true;
  }

  // Given X successes and n trials, calculate the interval using the
  // rho acceptance sets implemented above.
  void calculate(const double X, const double n) {
    set(0, 1);

    const double tol = 1e-9;
    double rho_min, rho_max, rho;
    int x_l, x_r;
  
    // Binary search for the smallest rho whose acceptance set has right
    // endpoint X; this is the lower endpoint of the rho interval.
    rho_min = 0; rho_max = 1;
    while (std::fabs(rho_max - rho_min) > tol) {
      rho = (rho_min + rho_max)/2;
      find_rho_set(rho, int(n), x_l, x_r);
      if (x_r < X)
	rho_min = rho;
      else
	rho_max = rho;
    }
    lower_ = rho;
  
    // Binary search for the largest rho whose acceptance set has left
    // endpoint X; this is the upper endpoint of the rho interval.
    rho_min = 0; rho_max = 1;
    while (std::fabs(rho_max - rho_min) > tol) {
      rho = (rho_min + rho_max)/2;
      find_rho_set(rho, int(n), x_l, x_r);
      if (x_l > X)
	rho_max = rho;
      else
	rho_min = rho;
    }
    upper_ = rho;
  }

 private:
  Sorter sorter_;

#if (defined (STANDALONE) or defined (__CINT__) )
ClassDefT(BinomialNoncentralInterval,1)
#endif
};

#endif
