#ifndef Utilities_NonNullNumberSelector_h
#define Utilities_NonNullNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: NonNullNumberSelector.h,v 1.1 2006/12/07 10:28:31 llista Exp $
 */

struct NonNullNumberSelector {
  NonNullNumberSelector() { }
  bool operator()( unsigned int number ) const { return number >= 0; }
};

#endif
