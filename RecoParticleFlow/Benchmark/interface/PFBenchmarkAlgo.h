#ifndef RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h
#define RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class PFBenchmarkAlgo {
public:

  // optional c'tor -- all methods can be used without an instance
  PFBenchmarkAlgo();
  virtual ~PFBenchmarkAlgo();

  // calculate base quantities for the given pair of candidates
  static double deltaEt(const reco::Candidate *, const reco::Candidate *);
  static double deltaEta(const reco::Candidate *, const reco::Candidate *);
  static double deltaPhi(const reco::Candidate *, const reco::Candidate *);
  static double deltaR(const reco::Candidate *, const reco::Candidate *);

  // simple candidate matching
  static const reco::Candidate *matchByDeltaEt(const reco::Candidate *, const reco::CandidateCollection *);
  static const reco::Candidate *matchByDeltaR(const reco::Candidate *, const reco::CandidateCollection *);

  // find a duplicate of the candidate in the collection and return a pointer to that element
  static const reco::Candidate *recoverCandidate(const reco::Candidate *, const reco::CandidateCollection *);

  // sorting functions - returns a sorted copy of the input
  static reco::CandidateCollection sortByDeltaR(const reco::Candidate *, const reco::CandidateCollection *);
  static reco::CandidateCollection sortByDeltaEt(const reco::Candidate *, const reco::CandidateCollection *);

  // multi-match function - returns a sorted, constrained collection
  static reco::CandidateCollection findAllInCone(const reco::Candidate *, const reco::CandidateCollection *, double ConeSize);
  static reco::CandidateCollection findAllInEtWindow(const reco::Candidate *, const reco::CandidateCollection *, double EtWindow);

};

#endif // RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h
