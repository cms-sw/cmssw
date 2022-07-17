#ifndef RecoAlgos_PtMaxSelector_h
#define RecoAlgos_PtMaxSelector_h
/* \class PtMaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMaxSelector.h,v 1.6 2007/06/18 18:33:54 llista Exp $
 */

struct PtMaxSelector {
  PtMaxSelector(double ptMax) : ptMax_(ptMax) {}
  template <typename T>
  bool operator()(const T& t) const {
    return t.pt() < ptMax_;
  }

private:
  double ptMax_;
};

#endif
