#ifndef CommonTools_Utils_MinNumberSelector_h
#define CommonTools_Utils_MinNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinNumberSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

struct MinNumberSelector {
  MinNumberSelector(unsigned int minNumber) : 
    minNumber_(minNumber) { }
  bool operator()(unsigned int number) const { return number >= minNumber_; }

private:
  unsigned int minNumber_;
};

#endif
