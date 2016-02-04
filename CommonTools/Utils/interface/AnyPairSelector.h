#ifndef CommonTools_Utils_AnyPairSelector_h
#define CommonTools_Utils_AnyPairSelector_h
/* \class AnyPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnyPairSelector.h,v 1.1 2009/02/24 14:10:18 llista Exp $
 */

struct AnyPairSelector {
  template<typename T1, typename T2>
  bool operator()( const T1 &, const T2 & ) const { return true; }
};

#endif
