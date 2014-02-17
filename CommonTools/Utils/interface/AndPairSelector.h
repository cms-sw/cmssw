#ifndef CommonTools_Utils_AndPairSelector_h
#define CommonTools_Utils_AndPairSelector_h
/* \class AndPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AndPairSelector.h,v 1.2 2012/06/26 21:13:11 wmtan Exp $
 */

template<typename S1, typename S2>
struct AndPairSelector {
  AndPairSelector(const S1 & s1, const S2 & s2) : s1_(s1), s2_(s2) { }
  template<typename T1, typename T2>
  bool operator()(const T1 & t1, const T2 & t2) const {
    return s1_(t1) && s2_(t2);
  }
private:
  S1 s1_;
  S2 s2_;
};

#endif
