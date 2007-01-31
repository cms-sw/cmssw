#ifndef RecoAlgos_RangeSelector_h
#define RecoAlgos_RangeSelector_h
/* \class RangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: RangeSelector.h,v 1.1 2006/10/03 11:45:06 llista Exp $
 */
#include <string>

template<typename T, double (T::*fun)() const>
struct RangeSelector {
  typedef T value_type;
  RangeSelector( double min, double max ) : 
    min_( min ), max_( max ) { }
  bool operator()( const value_type & t ) const { 
    double x = (t.*fun)();
    return min_ <= x && x <= max_; 
  }
private:
  double min_, max_;
};

#endif
