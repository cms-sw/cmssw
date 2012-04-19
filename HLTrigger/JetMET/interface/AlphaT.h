#ifndef HLTrigger_JetMET_AlphaT_h
#define HLTrigger_JetMET_AlphaT_h

#include <algorithm>
#include <functional>
#include <vector>


class AlphaT {
public:
  template <class T>
  AlphaT(std::vector<T const *> const & p4, bool use_et = true);

  template <class T>
  AlphaT(std::vector<T> const & p4, bool use_et = true);

private:
  // momentum in transverse plane
  std::vector<double> et_;
  std::vector<double> px_;
  std::vector<double> py_;

  // momentum sums in transverse plane
  double sum_et_;
  double denominator_;

public:
  inline double value(void) const;
  inline double value(std::vector<bool> & jet_sign) const;

  // return an approximate value of AlphaT
  // by construction, this shuld be always lower than, or equal to, the actual AlphaT value
  inline double approximate_value(void) const;
  inline double approximate_value(std::vector<bool> & jet_sign) const;

private:
  double value_(std::vector<bool> * jet_sign) const;
  double approximate_value_(std::vector<bool> * jet_sign) const;
};


// -----------------------------------------------------------------------------
static
bool greater_than(double a, double b) {
  return a > b;
}

// -----------------------------------------------------------------------------
template<class T>
AlphaT::AlphaT(std::vector<T const *> const & p4, bool use_et /* = true */) {
  // momentum in transverse plane
  std::transform( p4.begin(), p4.end(), back_inserter(et_), ( use_et ? std::mem_fun(&T::Et) : std::mem_fun(&T::Pt) ) );
  std::transform( p4.begin(), p4.end(), back_inserter(px_), std::mem_fun(&T::Px) );
  std::transform( p4.begin(), p4.end(), back_inserter(py_), std::mem_fun(&T::Py) );

  // momentum sums in transverse plane
  sum_et_ = std::accumulate( et_.begin(), et_.end(), 0. );
  double sum_px = std::accumulate( px_.begin(), px_.end(), 0. );
  double sum_py = std::accumulate( py_.begin(), py_.end(), 0. );
  denominator_ = 2. * sqrt( sum_et_ * sum_et_ - sum_px * sum_px - sum_py * sum_py);

  // make sure the ET vector is sorted in decreasing order
  std::sort(et_.begin(), et_.end(), greater_than);
}

// -----------------------------------------------------------------------------
template<class T>
AlphaT::AlphaT(std::vector<T> const & p4, bool use_et /* = true */) {
  // momentum in transverse plane
  std::transform( p4.begin(), p4.end(), back_inserter(et_), std::mem_fun_ref( use_et ? &T::Et : &T::Pt ) );
  std::transform( p4.begin(), p4.end(), back_inserter(px_), std::mem_fun_ref(&T::Px) );
  std::transform( p4.begin(), p4.end(), back_inserter(py_), std::mem_fun_ref(&T::Py) );

  // momentum sums in transverse plane
  sum_et_ = std::accumulate( et_.begin(), et_.end(), 0. );
  double sum_px = std::accumulate( px_.begin(), px_.end(), 0. );
  double sum_py = std::accumulate( py_.begin(), py_.end(), 0. );
  denominator_ = 2. * sqrt( sum_et_ * sum_et_ - sum_px * sum_px - sum_py * sum_py);

  // make sure the ET vector is sorted in decreasing order
  std::sort(et_.begin(), et_.end(), greater_than);
}

// -----------------------------------------------------------------------------
inline
double AlphaT::value(void) const {
  return value_(0);
}

// -----------------------------------------------------------------------------
inline
double AlphaT::value(std::vector<bool> & jet_sign) const {
  return value_(& jet_sign);
}

// -----------------------------------------------------------------------------
inline
double AlphaT::approximate_value(void) const {
  return approximate_value_(0);
}

// -----------------------------------------------------------------------------
inline
double AlphaT::approximate_value(std::vector<bool> & jet_sign) const {
  return approximate_value_(& jet_sign);
}

#endif // HLTrigger_JetMET_AlphaT_h
