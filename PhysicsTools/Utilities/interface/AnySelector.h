#ifndef Utilities_AnySelector_h
#define Utilities_AnySelector_h
/* \class AnySelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnySelector.h,v 1.3 2006/11/23 16:32:57 llista Exp $
 */

template<typename T>
struct AnySelector {
  typedef T value_type;
  bool operator()( const T & ) const { return true; }
};

#endif
