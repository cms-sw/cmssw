#ifndef Utilities_NonNullNumberSelector_h
#define Utilities_NonNullNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: NonNullNumberSelector.h,v 1.2 2007/01/31 14:42:59 llista Exp $
 */

struct NonNullNumberSelector {
  NonNullNumberSelector() { }
  bool operator()( unsigned int number ) const { return number > 0; }
};

#endif
