#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <cmath>
#include <algorithm>

using namespace reco;
using namespace std;

// functor for sorting candidatecollections by delta-r
class deltaRSorter: public binary_function<Candidate, Candidate, bool> {
public:

  deltaRSorter(const Candidate *Ref) { ref = Ref; }
  bool operator()(const Candidate &c1, const Candidate &c2) const {
    return PFBenchmarkAlgo::deltaR(ref,&c1) < PFBenchmarkAlgo::deltaR(ref,&c2);
  }

private:

  const Candidate *ref;

};

// functor for sorting candidatecollections by delta-et
class deltaEtSorter: public binary_function<Candidate, Candidate, bool> {
public:

  deltaEtSorter(const Candidate *Ref) { ref = Ref; }
  bool operator()(const Candidate &c1, const Candidate &c2) const {
    return PFBenchmarkAlgo::deltaEt(ref,&c1) < PFBenchmarkAlgo::deltaEt(ref,&c2);
  }

private:

  const Candidate *ref;

};


PFBenchmarkAlgo::PFBenchmarkAlgo() {}
PFBenchmarkAlgo::~PFBenchmarkAlgo() {}

void PFBenchmarkAlgo::reset() {

  // Delete the storage
  vector<CandidateCollection *>::iterator cc;
  for (cc = allocatedMem_.begin(); cc != allocatedMem_.end(); cc++)
    if (*cc != NULL) delete *cc;

  // Clear the container
  allocatedMem_.clear();

}

double PFBenchmarkAlgo::deltaEt(const Candidate *c1, const Candidate *c2) {

  return c1->et() - c2->et();

}

double PFBenchmarkAlgo::deltaPhi(const Candidate *c1, const Candidate *c2) {

  double phi1 = c1->phi();
  while (phi1 > M_PI)   phi1 -= 2 * M_PI;
  while (phi1 <= -M_PI) phi1 += 2 * M_PI;

  double phi2 = c2->phi();
  while (phi2 > M_PI)   phi2 -= 2 * M_PI;
  while (phi2 <= -M_PI) phi2 += 2 * M_PI;

  return phi1 - phi2;

}

double PFBenchmarkAlgo::deltaEta(const Candidate *c1, const Candidate *c2) {

  return c1->eta() - c2->eta();

}

double PFBenchmarkAlgo::deltaR(const Candidate *c1, const Candidate *c2) {

  double dphi = deltaPhi(c1,c2);
  double deta = deltaEta(c1,c2);

  return sqrt(pow(dphi,2) + pow(deta,2));

}

const Candidate *PFBenchmarkAlgo::matchByDeltaR(const Candidate *c1, const CandidateCollection *candidates) {

  double minDeltaR = 99.;
  const Candidate *bestMatch = NULL;

  // Loop Over the Candidates...
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {
 
    const Candidate *c2 = &(*candidate);

    // Find Minimal Delta-R
    double dR = deltaR(c1,c2);
    if (dR <= minDeltaR) {
      bestMatch = c2;
      minDeltaR = dR;
    }

  }

  return bestMatch;

}

const Candidate *PFBenchmarkAlgo::matchByDeltaEt(const Candidate *c1, const CandidateCollection *candidates) {

  double minDeltaEt = 9999.;
  const Candidate *bestMatch = NULL;

  // Loop Over the Candidates...
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {

    const Candidate *c2 = &(*candidate);

    // Find Minimal Delta-Et
    double dEt = fabs(deltaEt(c1,c2));
    if (dEt <= minDeltaEt) {
      bestMatch = c2;
      minDeltaEt = dEt;
    }

  }

  return bestMatch;

}

const Candidate *PFBenchmarkAlgo::recoverCandidate(const Candidate *c1, const CandidateCollection *candidates) {

  // Numerical Epsilon Factor for Comparing Equivalent Quantities
  double eps = 1e-6;

  // Loop Over the Candidates...
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {

    const Candidate *c2 = &(*candidate);

    // Candidates are assumed to be the same if eta, phi, and Et are approximately the same
    if (deltaR(c1,c2) < eps && deltaEt(c1,c2) < eps)
      return c2;

  }

  // Return NULL if the Candidate was Not Found in the Collection
  return NULL;

}

const CandidateCollection *PFBenchmarkAlgo::sortByDeltaR(const Candidate *c1, const CandidateCollection *candidates) {

  // allocate storage and store pointer for bookkeeping
  reco::CandidateCollection *sorted = new reco::CandidateCollection();
  allocatedMem_.push_back(sorted);

  // copy the input collection
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++)
    sorted->push_back((Candidate* const)candidate->clone());

  // sort by dR and return
  sorted->sort(deltaRSorter(c1));
  return reinterpret_cast<const CandidateCollection *>(sorted);
  
}

const CandidateCollection *PFBenchmarkAlgo::sortByDeltaEt(const Candidate *c1, const CandidateCollection *candidates) {

  // Allocate storage and store pointer for bookkeeping
  reco::CandidateCollection *sorted = new reco::CandidateCollection();
  allocatedMem_.push_back(sorted);

  // Copy the input Collection
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++)
    sorted->push_back((Candidate* const)candidate->clone());

  // Sort by dR and return
  sorted->sort(deltaEtSorter(c1));
  return reinterpret_cast<const CandidateCollection *>(sorted);

}

const CandidateCollection *PFBenchmarkAlgo::findAllInCone(const Candidate *c1, const CandidateCollection *candidates, double ConeSize) {

  // Allocate storage and store pointer for bookkeeping
  reco::CandidateCollection *inCone = new reco::CandidateCollection();
  allocatedMem_.push_back(inCone);

  // Copy the input Collection
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {
    
    const Candidate *c2 = &(*candidate);
    double dR = deltaR(c1,c2);
    if (dR < ConeSize) inCone->push_back((Candidate* const)c2->clone());

  }

  // Sort by dR and return
  inCone->sort(deltaRSorter(c1));
  return reinterpret_cast<const CandidateCollection *>(inCone);

}

const CandidateCollection *PFBenchmarkAlgo::findAllInEtWindow(const Candidate *c1, const CandidateCollection *candidates, double EtWindow) {

  // Allocate storage and store pointer for bookkeeping
  reco::CandidateCollection *inEtWindow = new reco::CandidateCollection();
  allocatedMem_.push_back(inEtWindow);

  // Copy the input Collection
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {
    
    const Candidate *c2 = &(*candidate);
    double dEt = fabs(deltaEt(c1,c2));
    if (dEt < EtWindow) inEtWindow->push_back((Candidate* const)c2->clone());

  }

  // Sort by dEt and return
  inEtWindow->sort(deltaEtSorter(c1));
  return reinterpret_cast<const CandidateCollection *>(inEtWindow);

}
