#ifndef Utilities_AnySelector_h
#define Utilities_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.4 2007/01/31 14:42:59 llista Exp $
 */

struct AnySelector {
  template<typename T>
  bool operator()( const T & ) const { return true; }
};

#endif
