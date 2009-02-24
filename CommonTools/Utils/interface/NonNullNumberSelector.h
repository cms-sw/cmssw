#ifndef CommonTools_Utils_NonNullNumberSelector_h
#define CommonTools_Utils_NonNullNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: NonNullNumberSelector.h,v 1.3 2007/06/07 08:59:43 llista Exp $
 */

struct NonNullNumberSelector {
  NonNullNumberSelector() { }
  bool operator()( unsigned int number ) const { return number > 0; }
};

#endif
