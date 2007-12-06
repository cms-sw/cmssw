#ifndef RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h
#define RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <vector>

class PFBenchmarkAlgo {
public:

  PFBenchmarkAlgo();
  virtual ~PFBenchmarkAlgo();

  // garbage management function
  // IMPORTANT -- run once at start or end of every analysis loop
  void reset();

  // calculate base quantities for the given pair of candidates
  static double deltaEt(const reco::Candidate *, const reco::Candidate *);
  static double deltaEta(const reco::Candidate *, const reco::Candidate *);
  static double deltaPhi(const reco::Candidate *, const reco::Candidate *);
  static double deltaR(const reco::Candidate *, const reco::Candidate *);

  // simple candidate matching
  const reco::Candidate *matchByDeltaEt(const reco::Candidate *, const reco::CandidateCollection *);
  const reco::Candidate *matchByDeltaR(const reco::Candidate *, const reco::CandidateCollection *);

  // find a duplicate of the candidate in the collection and return a pointer to that element
  const reco::Candidate *recoverCandidate(const reco::Candidate *, const reco::CandidateCollection *);

  // sorting functions - returns a sorted copy of the input
  const reco::CandidateCollection *sortByDeltaR(const reco::Candidate *, const reco::CandidateCollection *);
  const reco::CandidateCollection *sortByDeltaEt(const reco::Candidate *, const reco::CandidateCollection *);

  // multi-match function - returns a sorted, constrained collection
  const reco::CandidateCollection *findAllInCone(const reco::Candidate *, const reco::CandidateCollection *, double ConeSize);
  const reco::CandidateCollection *findAllInEtWindow(const reco::Candidate *, const reco::CandidateCollection *, double EtWindow);

  // make CandidateCollection out of vector<T> (for example, a PFCandidateCollection)
  template <typename CandidateDerived>
  const reco::CandidateCollection *makeCandidateCollection(const std::vector<CandidateDerived> *);

private:

  // keep track of what pointers have been allocated
  std::vector<reco::CandidateCollection *> allocatedMem_;

};

// template implementation (required to be in header)
template <typename CandidateDerived>
const reco::CandidateCollection *PFBenchmarkAlgo::makeCandidateCollection(const std::vector<CandidateDerived> *InputCollection) {

  reco::CandidateCollection *copy_candidates = new reco::CandidateCollection();
  allocatedMem_.push_back(copy_candidates);

  for (unsigned int i = 0; i < InputCollection->size(); i++) {
    CandidateDerived *c = (*InputCollection)[i].clone();
    copy_candidates->push_back((CandidateDerived* const)c);
  }

  return copy_candidates;

}

#endif // RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h
