#ifndef MatchByDRDPt_h_
#define MatchByDRDPt_h_

/** Define match between two objects by deltaR and deltaPt.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/MatchByDR.h"

namespace reco {
  template <typename T1, typename T2> class MatchByDRDPt {
  public:
    MatchByDRDPt (const edm::ParameterSet& cfg) :
      deltaR_(cfg),
      maxDPtRel_(cfg.getParameter<double>("maxDPtRel")) {}
    bool operator() (const T1& t1, const T2& t2) const {
      return fabs(t1.pt()-t2.pt())/t2.pt()<maxDPtRel_ &&
	deltaR_(t1,t2);
    }
  private:
    reco::MatchByDR<T1,T2> deltaR_;
    double maxDPtRel_;
  };
}


#endif
