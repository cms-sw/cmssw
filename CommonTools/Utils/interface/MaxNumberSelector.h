#ifndef CommonTools_Utils_MaxNumberSelector_h
#define CommonTools_Utils_MaxNumberSelector_h
/* \class SizeMaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxNumberSelector.h,v 1.1 2007/11/30 13:33:43 llista Exp $
 */

struct MaxNumberSelector {
  MaxNumberSelector(unsigned int maxNumber) : 
    maxNumber_(maxNumber) { }
  bool operator()(unsigned int number) const { return number <= maxNumber_; }

private:
  unsigned int maxNumber_;
};

#endif
