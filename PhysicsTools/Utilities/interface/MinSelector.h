#ifndef RecoAlgos_MinSelector_h
#define RecoAlgos_MinSelector_h
/* \class MinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinSelector.h,v 1.3 2007/01/31 14:42:59 llista Exp $
 */

template<typename T>
struct MinSelector {
  typedef T value_type;
  MinSelector( T min ) : min_( min ) { }
  bool operator()( T t ) const { return t >= min_; }

private:
  double min_;
};

#endif
