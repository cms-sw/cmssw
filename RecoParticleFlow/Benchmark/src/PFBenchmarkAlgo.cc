#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

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

    // Find Minimal Delta-R
    double dR = deltaR(c1,&(*candidate));
    if (dR <= minDeltaR) {
      bestMatch = &(*candidate);
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

    // Find Minimal Delta-Et
    double dEt = fabs(deltaEt(c1,&(*candidate)));
    if (dEt <= minDeltaEt) {
      bestMatch = &(*candidate);
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
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++)

    // Candidates are assumed to be the same if eta, phi, and Et are the same
    if (deltaR(c1,&(*candidate)) < eps && deltaEt(c1,&(*candidate)) < eps) return &(*candidate);

  // Return NULL if the Candidate was Not Found in the Collection
  return NULL;

}

CandidateCollection PFBenchmarkAlgo::sortByDeltaR(const Candidate *c1, const CandidateCollection *candidates) {

  CandidateCollection sorted = *candidates;
  sorted.sort(deltaRSorter(c1));

  // question: is the OwnVector c'tor sufficient, or should we copy using:
  /*CandidateCollection sorted;
  sorted.resize(candidates->size());
  copy(candidates->begin(),candidates->end(),sorted.begin());
  sort(sorted.begin(),sorted.end(),deltaRSorter());*/
  return sorted;
  
}

CandidateCollection PFBenchmarkAlgo::sortByDeltaEt(const Candidate *c1, const CandidateCollection *candidates) {

  CandidateCollection sorted = *candidates;
  //sort(sorted.begin(),sorted.end(),deltaEtSorter());
  sorted.sort(deltaRSorter(c1));
  return sorted;

}

CandidateCollection PFBenchmarkAlgo::findAllInCone(const Candidate *c1, const CandidateCollection *candidates, double ConeSize) {

  // Copy the input Collection
  CandidateCollection inCone = *candidates;

  // Loop Over the Candidates...
  CandidateCollection::iterator candidate;
  for (candidate = inCone.begin(); candidate != inCone.end(); candidate++) {

    // Remove Out-of-Cone Candidates from the copied Collection
    double dR = deltaR(c1,&(*candidate));
    if (dR >= ConeSize) inCone.erase(candidate);

  }

  // Sort and Return
  inCone.sort(deltaRSorter(c1));
  return inCone;

}

CandidateCollection PFBenchmarkAlgo::findAllInEtWindow(const Candidate *c1, const CandidateCollection *candidates, double EtWindow) {

  // Copy the input Collection
  CandidateCollection inWindow = *candidates;

  // Loop Over the Candidates...
  CandidateCollection::iterator candidate;
  for (candidate = inWindow.begin(); candidate != inWindow.end(); candidate++) {

    // Remove Out-of-Et Window Candidates from the copied Collection
    double dEt = fabs(deltaEt(c1,&(*candidate)));
    if (dEt >= EtWindow) inWindow.erase(candidate);

  }

  // Sort and Return
  inWindow.sort(deltaEtSorter(c1));
  return inWindow;

}
