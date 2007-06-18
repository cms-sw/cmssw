#ifndef Utilities_AnyPairSelector_h
#define Utilities_AnyPairSelector_h
/* \class AnyPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnyPairSelector.h,v 1.2 2007/01/31 14:42:59 llista Exp $
 */

struct AnyPairSelector {
  template<typename T1, typename T2>
  bool operator()( const T1 &, const T2 & ) const { return true; }
};

#endif
