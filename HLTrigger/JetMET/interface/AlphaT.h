#ifndef HLTrigger_JetMET_AlphaT_h
#define HLTrigger_JetMET_AlphaT_h

#include <algorithm>
#include <functional>
#include <vector>


class AlphaT {
public:
  template <class T>
  AlphaT(std::vector<T const *> const & p4, bool setDHtZero = false , bool use_et = true);

  template <class T>
  AlphaT(std::vector<T> const & p4, bool setDHtZero = false, bool use_et = true);

private:
  std::vector<double> et_;
  std::vector<double> px_;
  std::vector<double> py_;

public:
  inline double value(void) const;
  inline double value(std::vector<bool> & jet_sign) const;

private:
  double value_(std::vector<bool> * jet_sign) const;
  bool setDHtZero_;
};


// -----------------------------------------------------------------------------
template<class T>
AlphaT::AlphaT(std::vector<T const *> const & p4, bool setDHtZero, bool use_et /* = true */) {
  std::transform( p4.begin(), p4.end(), back_inserter(et_), ( use_et ? std::mem_fun(&T::Et) : std::mem_fun(&T::Pt) ) );
  std::transform( p4.begin(), p4.end(), back_inserter(px_), std::mem_fun(&T::Px) );
  std::transform( p4.begin(), p4.end(), back_inserter(py_), std::mem_fun(&T::Py) );
  setDHtZero_ = setDHtZero;
}

// -----------------------------------------------------------------------------
template<class T>
AlphaT::AlphaT(std::vector<T> const & p4, bool setDHtZero, bool use_et /* = true */) {
  std::transform( p4.begin(), p4.end(), back_inserter(et_), std::mem_fun_ref( use_et ? &T::Et : &T::Pt ) );
  std::transform( p4.begin(), p4.end(), back_inserter(px_), std::mem_fun_ref(&T::Px) );
  std::transform( p4.begin(), p4.end(), back_inserter(py_), std::mem_fun_ref(&T::Py) );
  setDHtZero_ = setDHtZero;
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

#endif // HLTrigger_JetMET_AlphaT_h
