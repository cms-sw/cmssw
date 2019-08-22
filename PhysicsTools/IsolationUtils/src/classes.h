#include "PhysicsTools/IsolationUtils/interface/PtIsolationAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef PtIsolationAlgo<reco::Candidate, reco::CandidateCollection> CandPtIsolationAlgo;

namespace PhysicsTools_IsolationUtils {
  struct dictionary {
    CandPtIsolationAlgo iso1;
  };
}  // namespace PhysicsTools_IsolationUtils
