#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include <cmath>
#include <algorithm>

using namespace reco;
using namespace std;

// Functor for sorting CandidateCollection inputs by Delta-R
class deltaRSorter: public binary_function<Candidate, Candidate, bool> {
public:

  deltaRSorter(const Candidate *Ref) { ref = Ref; }
  bool operator()(const Candidate &c1, const Candidate &c2) const {
    return PFBenchmarkAlgo::deltaR(ref,&c1) < PFBenchmarkAlgo::deltaR(ref,&c2);
  }

private:

  const Candidate *ref;

};

// Functor for sorting CandidateCollection inputs by Delta-Et
class deltaEtSorter: public binary_function<Candidate, Candidate, bool> {
public:

  deltaEtSorter(const Candidate *Ref) { ref = Ref; }
  bool operator()(const Candidate &c1, const Candidate &c2) const {
    return PFBenchmarkAlgo::deltaEt(ref,&c1) < PFBenchmarkAlgo::deltaEt(ref,&c2);
  }

private:

  const Candidate *ref;

};


double PFBenchmarkAlgo::deltaEt(const Candidate *c1, const Candidate *c2) {

  // Verify the valididy of Candidates
  if (!validCandidate(c1) || !validCandidate(c2))
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaEt for invalid Candidate(s)";

  // Calculate dEt (may be negative, usually called with reco - true)
  return c1->et() - c2->et();

}

double PFBenchmarkAlgo::deltaPhi(const Candidate *c1, const Candidate *c2) {

  // Verify the valididy of Candidates
  if (!validCandidate(c1) || !validCandidate(c2))
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaPhi for invalid Candidate(s)";

  double phi1 = c1->phi();
  while (phi1 > M_PI)   phi1 -= 2 * M_PI;
  while (phi1 <= -M_PI) phi1 += 2 * M_PI;

  double phi2 = c2->phi();
  while (phi2 > M_PI)   phi2 -= 2 * M_PI;
  while (phi2 <= -M_PI) phi2 += 2 * M_PI;

  // Calculate dPhi (may be negative, usually called with reco - true)
  return phi1 - phi2;

}

double PFBenchmarkAlgo::deltaEta(const Candidate *c1, const Candidate *c2) {

  // Verify the valididy of Candidates
  if (!validCandidate(c1) || !validCandidate(c2))
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaEta for invalid Candidate(s)";

  // Calculate dEta (may be negative, usually called with reco - true)
  return c1->eta() - c2->eta();

}

double PFBenchmarkAlgo::deltaR(const Candidate *c1, const Candidate *c2) {

  // Verify the valididy of Candidates
  if (!validCandidate(c1) || !validCandidate(c2))
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaR for invalid Candidate(s)";

  double dphi = deltaPhi(c1,c2);
  double deta = deltaEta(c1,c2);

  // Calculate dR
  return sqrt(pow(dphi,2) + pow(deta,2));

}

const Candidate *PFBenchmarkAlgo::matchByDeltaR(const Candidate *c1, const CandidateCollection *candidates) {

  // Verify the valididy of Candidate and the CandidateCollection
  if (!validCandidate(c1))
    throw cms::Exception("Invalid Arg") << "attempted to match invalid Candidate";
  if (!validCandidateCollection(candidates))
    throw cms::Exception("Invalid Arg") << "attempted to match to invalid CandidateCollection";

  double minDeltaR = 9999.;
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

  // Return the Candidate with the smallest dR
  return bestMatch;

}

const Candidate *PFBenchmarkAlgo::matchByDeltaEt(const Candidate *c1, const CandidateCollection *candidates) {

  // Verify the valididy of Candidate and the CandidateCollection
  if (!validCandidate(c1))
    throw cms::Exception("Invalid Arg") << "attempted to match invalid Candidate";
  if (!validCandidateCollection(candidates))
    throw cms::Exception("Invalid Arg") << "attempted to match to invalid CandidateCollection";

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

  // Return the Candidate with the smallest fabs(dEt)
  return bestMatch;

}

const Candidate *PFBenchmarkAlgo::recoverCandidate(const Candidate *c1, const CandidateCollection *candidates) {

  // Numerical epsilon Factor for Comparing Equivalent Quantities
  double eps = 1e-6;

  // Loop Over the Candidates...
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {

    const Candidate *c2 = &(*candidate);

    // Candidates are assumed to be the same if eta, phi, and Et are the same
    if (deltaR(c1,c2) < eps && deltaEt(c1,c2) < eps) return c2;

  }

  // Return NULL if the Candidate was not found in the Collection
  return NULL;

}

CandidateCollection PFBenchmarkAlgo::sortByDeltaR(const Candidate *c1, const CandidateCollection *candidates) {

  // Verify the valididy of Candidate and the CandidateCollection
  if (!validCandidate(c1))
    throw cms::Exception("Invalid Arg") << "attempted to sort by invalid Candidate";
  if (!validCandidateCollection(candidates))
    throw cms::Exception("Invalid Arg") << "attempted to sort invalid CandidateCollection";

  reco::CandidateCollection sorted;

  // Copy the input Collection
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {
    const Candidate *c2 = &(*candidate);
    sorted.push_back((Candidate* const)c2->clone());
  }

  // Sort and return Collection
  sorted.sort(deltaRSorter(c1));
  return sorted;
  
}

CandidateCollection PFBenchmarkAlgo::sortByDeltaEt(const Candidate *c1, const CandidateCollection *candidates) {

  // Verify the valididy of Candidate and the CandidateCollection
  if (!validCandidate(c1))
    throw cms::Exception("Invalid Arg") << "attempted to sort by invalid Candidate";
  if (!validCandidateCollection(candidates))
    throw cms::Exception("Invalid Arg") << "attempted to sort invalid CandidateCollection";

  reco::CandidateCollection sorted;
  
  // Copy the input Collection
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {
    const Candidate *c2 = &(*candidate);
    sorted.push_back((Candidate* const)c2->clone());
  }

  // Sort and return Collection
  sorted.sort(deltaEtSorter(c1));
  return sorted;

}

CandidateCollection PFBenchmarkAlgo::findAllInCone(const Candidate *c1, const CandidateCollection *candidates, double ConeSize) {

  // Verify the valididy of Candidate and the CandidateCollection
  if (!validCandidate(c1))
    throw cms::Exception("Invalid Arg") << "attempted to sort by invalid Candidate";
  if (!validCandidateCollection(candidates))
    throw cms::Exception("Invalid Arg") << "attempted to sort invalid CandidateCollection";
  if (ConeSize <= 0)
    throw cms::Exception("Invalid Arg") << "zero or negative cone size specified";

  reco::CandidateCollection constrained;

  // Copy the input Collection 
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {

    const Candidate *c2 = &(*candidate);

    // Add in-cone Candidates to the new Collection
    double dR = deltaR(c1,c2);
    if (dR < ConeSize) constrained.push_back((Candidate* const)c2->clone());

  }

  // Sort and return Collection
  constrained.sort(deltaRSorter(c1));
  return constrained;

}

CandidateCollection PFBenchmarkAlgo::findAllInEtWindow(const Candidate *c1, const CandidateCollection *candidates, double EtWindow) {

  // Verify the valididy of Candidate and the CandidateCollection
  if (!validCandidate(c1))
    throw cms::Exception("Invalid Arg") << "attempted to sort by invalid Candidate";
  if (!validCandidateCollection(candidates))
    throw cms::Exception("Invalid Arg") << "attempted to sort invalid CandidateCollection";
  if (EtWindow <= 0)
    throw cms::Exception("Invalid Arg") << "zero or negative cone size specified";

  reco::CandidateCollection constrained;

  // Copy the input Collection 
  CandidateCollection::const_iterator candidate;
  for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {

    const Candidate *c2 = &(*candidate);

    // Add in-Et-Window Candidates to the new Collection
    double dEt = fabs(deltaEt(c1,c2));
    if (dEt < EtWindow) constrained.push_back((Candidate* const)c2->clone());

  }

  // Sort and return Collection
  constrained.sort(deltaEtSorter(c1));
  return constrained;

}

bool PFBenchmarkAlgo::validCandidate(const Candidate *c) {

  return c == NULL ? false : true;

}

bool PFBenchmarkAlgo::validPFCandidate(const Candidate *c) {

  const PFCandidate *pfcand = dynamic_cast<const PFCandidate *>(c);
  return pfcand == NULL ? false : true;

}

bool PFBenchmarkAlgo::validPFJet(const Candidate *c) {

  const PFJet *pfjet = dynamic_cast<const PFJet *>(c);
  return pfjet == NULL ? false : true;

}

bool PFBenchmarkAlgo::validCaloJet(const Candidate *c) {

  const CaloJet *calojet = dynamic_cast<const CaloJet *>(c);
  return calojet == NULL ? false : true;

}

bool PFBenchmarkAlgo::validCandidateCollection(const CandidateCollection *candidates) {

  return candidates == NULL ? false : true;

}
