#ifndef Utilities_MinNumberSelector_h
#define Utilities_MinNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinNumberSelector.h,v 1.4 2007/01/31 14:42:59 llista Exp $
 */

struct MinNumberSelector {
  MinNumberSelector(unsigned int minNumber) : 
    minNumber_(minNumber) { }
  bool operator()(unsigned int number) const { return number >= minNumber_; }

private:
  unsigned int minNumber_;
};

#endif
