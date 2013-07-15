#ifndef CommonTools_Utils_MinNumberSelector_h
#define CommonTools_Utils_MinNumberSelector_h
/* \class SizeMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinNumberSelector.h,v 1.5 2007/11/30 13:33:43 llista Exp $
 */

struct MinNumberSelector {
  MinNumberSelector(unsigned int minNumber) : 
    minNumber_(minNumber) { }
  bool operator()(unsigned int number) const { return number >= minNumber_; }

private:
  unsigned int minNumber_;
};

#endif
