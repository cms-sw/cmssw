#include "PhysicsTools/IsolationUtils/interface/PtIsolationAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef PtIsolationAlgo<reco::Candidate,reco::CandidateCollection> CandPtIsolationAlgo;

namespace {
  struct dictionary {
    CandPtIsolationAlgo iso1;
  };
}
