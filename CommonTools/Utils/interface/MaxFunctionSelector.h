#ifndef RecoAlgos_MaxFunctionSelector_h
#define RecoAlgos_MaxFunctionSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxFunctionSelector.h,v 1.2 2012/06/26 21:13:12 wmtan Exp $
 */

template<typename T, double (T::*fun)() const>
struct MaxFunctionSelector {
  MaxFunctionSelector( double max ) : 
    max_( max ) { }
  bool operator()( const T & t ) const { return (t.*fun)() <= max_; }

private:
  double max_;
};

#endif
