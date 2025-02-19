#ifndef CommonTools_Utils_NonNullNumberSelector_h
#define CommonTools_Utils_NonNullNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: NonNullNumberSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

struct NonNullNumberSelector {
  NonNullNumberSelector() { }
  bool operator()( unsigned int number ) const { return number > 0; }
};

#endif
