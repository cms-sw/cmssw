#include "PhysicsTools/IsolationUtils/interface/PtIsolationAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef PtIsolationAlgo<reco::Candidate,reco::CandidateCollection> CandPtIsolationAlgo;

namespace {
  namespace {
    CandPtIsolationAlgo iso1;
  }
}
