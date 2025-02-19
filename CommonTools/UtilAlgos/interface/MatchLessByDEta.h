#ifndef MatchLessByDEta_h_
#define MatchLessByDEta_h_

/** Provides the less operator for two pairs of matched objects based
 *  on deltaEta. */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  template <typename C1, typename C2> struct MatchLessByDEta {
  public: 
    MatchLessByDEta ( const edm::ParameterSet& cfg,
		     const C1& c1, const C2& c2) :
      c1_(c1), c2_(c2) {}
    bool operator() (const std::pair<size_t,size_t>& p1,
		     const std::pair<size_t,size_t>& p2) const {
      typedef typename C1::value_type T1;
      typedef typename C2::value_type T2;
      const T1& p1_1 = c1_[p1.first];
      const T2& p1_2 = c2_[p1.second];
      const T1& p2_1 = c1_[p2.first];
      const T2& p2_2 = c2_[p2.second];
      if ( fabs(p1_1.eta()-p1_2.eta()) <
	   fabs(p2_1.eta()-p2_2.eta()) )  return true;
      return false;
    }
  private:
    const C1& c1_;
    const C2& c2_;
  };
}

#endif
