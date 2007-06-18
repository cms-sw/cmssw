#ifndef RecoAlgos_MinSelector_h
#define RecoAlgos_MinSelector_h
/* \class MinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinSelector.h,v 1.4 2007/05/15 16:07:53 llista Exp $
 */

template<typename T>
struct MinSelector {
  MinSelector( T min ) : min_( min ) { }
  bool operator()( T t ) const { return t >= min_; }

private:
  T min_;
};

#endif
