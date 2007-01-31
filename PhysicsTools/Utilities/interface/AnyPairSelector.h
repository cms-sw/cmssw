#ifndef Utilities_AnyPairSelector_h
#define Utilities_AnyPairSelector_h
/* \class AnyPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnyPairSelector.h,v 1.1 2006/11/23 16:32:57 llista Exp $
 */

template<typename T>
struct AnyPairSelector {
  typedef T value_type;
  bool operator()( const T &, const T & ) const { return true; }
};

#endif
