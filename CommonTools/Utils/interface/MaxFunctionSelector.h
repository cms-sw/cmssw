#ifndef RecoAlgos_MaxFunctionSelector_h
#define RecoAlgos_MaxFunctionSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxFunctionSelector.h,v 1.2 2007/06/18 18:33:53 llista Exp $
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
