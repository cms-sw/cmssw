#ifndef CommonTools_Utils_AnySelector_h
#define CommonTools_Utils_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.5 2007/06/18 18:33:53 llista Exp $
 */

struct AnySelector {
  template<typename T>
  bool operator()( const T & ) const { return true; }
};

#endif
