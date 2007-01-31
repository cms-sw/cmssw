#ifndef RecoAlgos_MinObjectPairSelector_h
#define RecoAlgos_MinObjectPairSelector_h
/* \class MinObjectPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinObjectPairSelector.h,v 1.1 2006/10/03 12:06:40 llista Exp $
 */

template<typename T, typename F>
struct MinObjectPairSelector {
  typedef T value_type;
  MinObjectPairSelector( double min ) : 
    min_( min ), fun_() { }
  bool operator()( const value_type & t1, const value_type & t2 ) const { 
    return min_ <= fun_( t1, t2 ); 
  }

private:
  double min_;
  F fun_;
};

#endif
