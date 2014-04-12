#ifndef RecoAlgos_RangeObjectPairSelector_h
#define RecoAlgos_RangeObjectPairSelector_h
/* \class RangeObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: RangeObjectPairSelector.h,v 1.4 2007/06/18 18:33:54 llista Exp $
 */

template<typename F>
struct RangeObjectPairSelector {
  typedef F function;
  RangeObjectPairSelector( double min, double max, const F & fun ) : 
    min_( min ), max_( max ), fun_( fun ) { }
  RangeObjectPairSelector( double min, double max ) : 
    min_( min ), max_( max ), fun_() { }
  template<typename T1, typename T2>
  bool operator()( const T1 & t1, const T2 & t2 ) const { 
    double x = fun_( t1, t2 );
    return ( min_ <= x && x <= max_ ); 
  }

private:
  double min_, max_;
  F fun_;
};

#endif
