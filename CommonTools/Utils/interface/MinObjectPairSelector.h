#ifndef RecoAlgos_MinObjectPairSelector_h
#define RecoAlgos_MinObjectPairSelector_h
/* \class MinObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinObjectPairSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

template<typename F>
struct MinObjectPairSelector {
  MinObjectPairSelector( double min ) : 
    min_( min ), fun_() { }
  template<typename T1, typename T2>
  bool operator()( const T1 & t1, const T2 & t2 ) const { 
    return min_ <= fun_( t1, t2 ); 
  }

private:
  double min_;
  F fun_;
};

#endif
