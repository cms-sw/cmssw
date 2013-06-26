#ifndef PhysicsTools_RooStatsCms_BinomialInterval_h
#define PhysicsTools_RooStatsCms_BinomialInterval_h
/* \class BinomialInterval
 *
 * \author Jordan Tucker
 * integration in CMSSW: Luca Lista
 *
 */


#if (defined (STANDALONE) or defined (__CINT__) )
#include "TNamed.h"
#endif

// A class to implement the calculation of intervals for the binomial
// parameter rho. The bulk of the work is done by derived classes that
// implement calculate() appropriately.
class BinomialInterval 
#if (defined (STANDALONE) or defined (__CINT__) )
: public TNamed
#endif
{
public:
  // For central intervals, an enum to indicate whether the interval
  // should put all of alpha at the lower or upper end, or equally
  // divide it.
  enum tail_type { equal_tailed, lower_tailed, upper_tailed };

  // Set alpha, type, and the cached values of kappa (the normal
  // quantile).
  void init(const double alpha, const tail_type t=equal_tailed);

  // Methods which derived classes must implement.

  // Calculate the interval given a number of successes and a number
  // of trials. (Successes/trials are doubles to having to cast to
  // double later anyway.)
  virtual void calculate(const double successes, const double trials) = 0;

  // Return a pretty name for the interval (e.g. "Feldman-Cousins").
  virtual const char* name() const = 0;

  // A simple test (depending on the tail type) whether a certain
  // value of the binomial parameter rho is in the interval or not.
  bool contains(double rho);
  
  // Calculate and return the coverage probability given the true
  // binomial parameter rho and the number of trials.
  double coverage_prob(const double rho, const int trials);

  // Convenience methods to scan the parameter space. In each, the
  // pointers must point to already-allocated arrays of double, with
  // the number of doubles depending on the method. (double is used
  // even for parameters that are naturally int for compatibility with
  // TGraph/etc.)

  // Given ntot trials, scan rho in [0,1] with nrho points (so the
  // arrays must be allocated double[nrho]).
  void scan_rho(const int ntot, const int nrho, double* rho, double* prob);

  // Given the true value of rho, scan over from ntot_min to ntot_max
  // trials (so the arrays must be allocated double[ntot_max -
  // ntot_min + 1]).
  void scan_ntot(const double rho, const int ntot_min, const int ntot_max, double* ntot, double* prob);

  // Construct nrho acceptance sets in rho = [0,1] given ntot trials
  // and put the results in x_l and x_r. The arrays must be allocated
  // as double[nrho].
  virtual bool neyman(const int ntot, const int nrho, double* rho, double* x_l, double* x_r) { return false; } 

  // Dump a table of intervals from trials_min to trials_max, with
  // successes = 0 to trials for each. The table is produced in a file
  // in the current directory, given the name of the interval.
  void dump(const int trials_min, const int trials_max);

  // Simple accessors.
  double alpha() const { return alpha_; }
  double lower() const { return lower_; }
  double upper() const { return upper_; }
  double length() const { return upper_ - lower_; }

protected:
  double    alpha_;
  tail_type type_;
  double    alpha_min_;
  double    kappa_;
  double    kappa2_;

  double lower_;
  double upper_;

  void set(double l, double u) { lower_ = l; upper_ = u; }

#if (defined (STANDALONE) or defined (__CINT__) )
ClassDef(BinomialInterval,1)
#endif
};

#endif

