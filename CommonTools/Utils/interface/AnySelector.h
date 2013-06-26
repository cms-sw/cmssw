#ifndef CommonTools_Utils_AnySelector_h
#define CommonTools_Utils_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.1 2009/02/24 14:10:19 llista Exp $
 */

struct AnySelector {
  template<typename T>
  bool operator()( const T & ) const { return true; }
};

#endif
