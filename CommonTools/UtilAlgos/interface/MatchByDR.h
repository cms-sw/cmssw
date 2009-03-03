#ifndef MatchByDR_h_
#define MatchByDR_h_

/** Define match between two objects by deltaR and deltaPt.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"

namespace reco {
  template <typename T1, typename T2> class MatchByDR {
  public:
    MatchByDR (const edm::ParameterSet& cfg) :
      maxDR_(cfg.getParameter<double>("maxDeltaR")) {}
    bool operator() (const T1& t1, const T2& t2) const {
      return deltaR_(t1,t2)<maxDR_;
    }
  private:
    DeltaR<T1,T2> deltaR_;
    double maxDR_;
  };
}


#endif
