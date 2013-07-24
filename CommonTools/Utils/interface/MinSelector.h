#ifndef RecoAlgos_MinSelector_h
#define RecoAlgos_MinSelector_h
/* \class MinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinSelector.h,v 1.2 2012/06/26 21:13:12 wmtan Exp $
 */

template<typename T>
struct MinSelector {
  MinSelector( T min ) : min_( min ) { }
  bool operator()( T t ) const { return t >= min_; }

private:
  T min_;
};

#endif
