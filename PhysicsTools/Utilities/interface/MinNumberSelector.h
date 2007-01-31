#ifndef Utilities_MinNumberSelector_h
#define Utilities_MinNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinNumberSelector.h,v 1.3 2006/12/08 09:41:39 llista Exp $
 */

struct MinNumberSelector {
  MinNumberSelector( unsigned int minNumber ) : 
    minNumber_( minNumber ) { }
  bool operator()( unsigned int number ) const { return number >= minNumber_; }

private:
  unsigned int minNumber_;
};

#endif
