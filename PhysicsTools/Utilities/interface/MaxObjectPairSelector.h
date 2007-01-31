#ifndef RecoAlgos_MaxObjectPairSelector_h
#define RecoAlgos_MaxObjectPairSelector_h
/* \class MaxObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxObjectPairSelector.h,v 1.1 2006/10/03 12:06:40 llista Exp $
 */

template<typename T, typename F>
struct MaxObjectPairSelector {
  typedef T value_type;
  MaxObjectPairSelector( double max ) : 
    max_( max ), fun_() { }
  bool operator()( const value_type & t1, const value_type & t2 ) const { 
    return fun_( t1, t2 ) <= max_;
  }

private:
  double max_;
  F fun_;
};

#endif
