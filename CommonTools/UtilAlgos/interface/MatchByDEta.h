#ifndef MatchByDEta_h_
#define MatchByDEta_h_

/** Define match between two objects by deltaEta.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  template <typename T1, typename T2> class MatchByDEta {
  public:
    MatchByDEta (const edm::ParameterSet& cfg) :
      maxDEta_(cfg.getParameter<double>("maxDeltaEta")) {}
    bool operator() (const T1& t1, const T2& t2) const {
      return fabs(t1.eta()-t2.eta()) < maxDEta_;
    }
  private:
    double maxDEta_;
  };
}


#endif
