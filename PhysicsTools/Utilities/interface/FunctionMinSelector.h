#ifndef Utilities_FunctionMinSelector_h
#define Utilities_FunctionMinSelector_h
/* \class FunctionMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: FunctionMinSelector.h,v 1.5 2007/06/18 18:33:53 llista Exp $
 */

template<typename F>
struct FunctionMinSelector {
  explicit FunctionMinSelector( double minCut ) :
    minCut_( minCut ) { }
  bool operator()( const typename F::type & t ) const {
    return f( t ) >= minCut_;
  }
private:
  F f;
  double minCut_;
};

#endif
