#ifndef Utilities_MaxNumberSelector_h
#define Utilities_MaxNumberSelector_h
/* \class SizeMaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxNumberSelector.h,v 1.4 2007/01/31 14:42:59 llista Exp $
 */

struct MaxNumberSelector {
  MaxNumberSelector(unsigned int maxNumber) : 
    maxNumber_(maxNumber) { }
  bool operator()(unsigned int number) const { return number <= maxNumber_; }

private:
  unsigned int maxNumber_;
};

#endif
