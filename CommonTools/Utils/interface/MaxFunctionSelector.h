#ifndef RecoAlgos_MaxFunctionSelector_h
#define RecoAlgos_MaxFunctionSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxFunctionSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
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
