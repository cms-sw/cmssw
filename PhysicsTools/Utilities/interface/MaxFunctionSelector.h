#ifndef RecoAlgos_MaxFunctionSelector_h
#define RecoAlgos_MaxFunctionSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.3 2007/01/31 14:42:59 llista Exp $
 */

template<typename T, double (T::*fun)() const>
struct MaxFunctionSelector {
  typedef T value_type;
  MaxFunctionSelector( double max ) : 
    max_( max ) { }
  bool operator()( const value_type & t ) const { return (t.*fun)() <= max_; }

private:
  double max_;
};

#endif
