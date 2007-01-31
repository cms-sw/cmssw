#ifndef RecoAlgos_RangeObjectPairSelector_h
#define RecoAlgos_RangeObjectPairSelector_h
/* \class RangeObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: RangeObjectPairSelector.h,v 1.2 2006/10/03 11:36:10 llista Exp $
 */

template<typename T, typename F>
struct RangeObjectPairSelector {
  typedef T value_type;
  typedef F function;
  RangeObjectPairSelector( double min, double max, const F & fun ) : 
    min_( min ), max_( max ), fun_( fun ) { }
  RangeObjectPairSelector( double min, double max ) : 
    min_( min ), max_( max ), fun_() { }
  bool operator()( const value_type & t1, const value_type & t2 ) const { 
    double x = fun_( t1, t2 );
    return ( min_ <= x && x <= max_ ); 
  }

private:
  double min_, max_;
  F fun_;
};

#endif
