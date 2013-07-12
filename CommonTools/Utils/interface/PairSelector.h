#ifndef CommonTools_Utils_PairSelector_h
#define CommonTools_Utils_PairSelector_h
/* \class PairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PairSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

template<typename S1, typename S2>
struct PairSelector {
  PairSelector(const S1 & s1, const S2 & s2) : s1_(s1), s2_(s2) { }
  template<typename T>
  bool operator()(const T & t) const {
    return s1_(t.first) && s2_(t.second);
  }
private:
  S1 s1_;
  S2 s2_;
};

#endif
