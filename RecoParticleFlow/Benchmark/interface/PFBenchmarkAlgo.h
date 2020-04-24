#ifndef RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h
#define RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include <cmath>
#include <vector>

// Notes on template implementation:
// - T, U are arbitrary types (must have et(), eta(), etc. defined)
//   support for Candidate-derived objects is explicit
// - Collection is a generic container class. Support for edm::OwnVector
//   and std::vector is explicit.

class PFBenchmarkAlgo {
public:

  // Calculate Delta-Et for the pair of Candidates (T - U)
  template <typename T, typename U>
  static double deltaEt(const T *, const U *);

  // Calculate Delta-Eta for the pair of Candidates (T - U)
  template <typename T, typename U>
  static double deltaEta(const T *, const U *);

  // Calculate Delta-Phi for the pair of Candidates (T - U)
  template <typename T, typename U>
  static double deltaPhi(const T *, const U *);

  // Calculate Delta-R for the pair of Candidates 
  template <typename T, typename U>
  static double deltaR(const T *, const U *);

  // Match Candidate T to a Candidate in the Collection based on minimum Delta-R
  template <typename T, typename Collection>
  static const typename Collection::value_type *matchByDeltaR(const T *, const Collection *);
  
  // Match Candidate T to a Candidate U in the Collection based on minimum Delta-Et
  template <typename T, typename Collection>
  static const typename Collection::value_type *matchByDeltaEt(const T *, const Collection *);

  // Copy the input Collection (useful when sorting)
  template <typename T, typename Collection>
  static Collection copyCollection(const Collection *);

  // Sort the U Candidates to the T Candidate based on minimum Delta-R
  template <typename T, typename Collection>
  static void sortByDeltaR(const T *, Collection &);

  // Sort the U Candidates to the T Candidate based on minimum Delta-Et
  template <typename T, typename Collection>
  static void sortByDeltaEt(const T *, Collection &);

  // Constrain the U Candidates to the T Candidate based on Delta-R to T
  template <typename T, typename Collection>
  static Collection findAllInCone(const T *, const Collection *, double);

  // Constrain the U Candidates to the T Candidate based on Delta-Et to T
  template <typename T, typename Collection>
  static Collection findAllInEtWindow(const T *, const Collection *, double);

private:

  // std::vector sort helper function
  template <typename T, typename U, template <typename,typename> class Sorter>
  static void vector_sort(std::vector<T> &candidates, Sorter<T,U> S) {
    sort(candidates.begin(),candidates.end(),S);
  }

  // std::vector push_back helper function
  template <typename T>
  static void vector_add(const T *c1, std::vector<T> &candidates) {
    candidates.push_back(*c1);
  }

  // edm::OwnVector sort helper functions
  template <typename T, typename U, template <typename,typename> class Sorter>
  static void vector_sort(edm::OwnVector<T> &candidates, Sorter<T,U> S) {
    candidates.sort(S);
  }

  // edm::OwnVector push_back helper function
  template <typename T>
  static void vector_add(const T *c1, edm::OwnVector<T> &candidates) {
    candidates.push_back((T *const)c1->clone());
  }

};

// ========================================================================
// implementation follows (required to be in header for templates)
// ========================================================================

// Helper class for sorting U Collections by Delta-R to a Candidate T
template <typename T, typename U> class deltaRSorter {
public:

  deltaRSorter(const T *Ref) { cref = Ref; }
  bool operator()(const U &c1, const U &c2) const {
    return PFBenchmarkAlgo::deltaR(cref,&c1) < PFBenchmarkAlgo::deltaR(cref,&c2);
  }

private:

  const T *cref;

};

// Helper class for sorting U Collections by Delta-Et to a Candidate T
template <typename T, typename U> class deltaEtSorter {
public:

  deltaEtSorter(const T *Ref) { cref = Ref; }
  bool operator()(const U &c1, const U &c2) const {
    return fabs(PFBenchmarkAlgo::deltaEt(cref,&c1)) < fabs(PFBenchmarkAlgo::deltaEt(cref,&c2));
  }

private:

  const T *cref;

};

// Calculate Delta-Et for Candidates (T - U)
template <typename T, typename U>
double PFBenchmarkAlgo::deltaEt(const T *c1, const U *c2) {

  if (c1 == nullptr || c2 == nullptr)
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaEt for invalid Candidate(s)";

  return c1->et() - c2->et();

}

// Calculate Delta-Eta for Candidates (T - U)
template <typename T, typename U>
double PFBenchmarkAlgo::deltaEta(const T *c1, const U *c2) {

  if (c1 == nullptr || c2 == nullptr)
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaEta for invalid Candidate(s)";

  return c1->eta() - c2->eta();

}

// Calculate Delta-Phi for Candidates (T - U)
template <typename T, typename U>
double PFBenchmarkAlgo::deltaPhi(const T *c1, const U *c2) {

  if (c1 == nullptr || c2 == nullptr)
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaPhi for invalid Candidate(s)";

  
  double phi1 = c1->phi();
  if (phi1 > M_PI) phi1 -= ceil((phi1 - M_PI) / (2 * M_PI)) * 2 * M_PI;
  if (phi1 <= - M_PI) phi1 += ceil((phi1 + M_PI) / (-2. * M_PI)) * 2. * M_PI;

  double phi2 = c2->phi();
  if (phi2 > M_PI) phi2 -= ceil((phi2 - M_PI) / (2 * M_PI)) * 2 * M_PI;
  if (phi2 <= - M_PI) phi2 += ceil((phi2 + M_PI) / (-2. * M_PI)) * 2 * M_PI;

  // alternative method:
  // while (phi > M_PI) phi -= 2 * M_PI;
  // while (phi <= - M_PI) phi += 2 * M_PI;

  double deltaphi=-999.0;
  if (fabs(phi1 - phi2)<M_PI)
  {
    deltaphi=(phi1-phi2);
  }
  else
  {
    if ((phi1-phi2)>0.0)
    {
      deltaphi=(2*M_PI - fabs(phi1 - phi2));
    }
    else
    {
      deltaphi=-(2*M_PI - fabs(phi1 - phi2));
    }
  }
  return deltaphi;
  //return ( (fabs(phi1 - phi2)<M_PI)?(phi1-phi2):(2*M_PI - fabs(phi1 - phi2) ) ); // FL: wrong
}
 
// Calculate Delta-R for Candidates 
template <typename T, typename U>
double PFBenchmarkAlgo::deltaR(const T *c1, const U *c2) {

  if (c1 == nullptr || c2 == nullptr)
    throw cms::Exception("Invalid Arg") << "attempted to calculate deltaR for invalid Candidate(s)";

  return sqrt(std::pow(deltaPhi(c1,c2),2) + std::pow(deltaEta(c1,c2),2));

}

// Match Candidate T to a Candidate in the Collection based on minimum Delta-R
template <typename T, typename Collection>
const typename Collection::value_type *PFBenchmarkAlgo::matchByDeltaR(const T *c1, const Collection *candidates) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of the Candidate and Collection
  if (!c1) throw cms::Exception("Invalid Arg") << "attempted to match invalid Candidate";
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to match to invalid Collection";

  double minDeltaR = 9999.;
  const U *match = nullptr;
  
  // Loop Over the Candidates...
  for (unsigned int i = 0; i < candidates->size(); i++) {

    const U *c2 = &(*candidates)[i];
    if (!c2) throw cms::Exception("Invalid Arg") << "attempted to match to invalid Candidate";

    // Find Minimal Delta-R
    double dR = deltaR(c1,c2);
    if (dR <= minDeltaR) { match = c2; minDeltaR = dR; }
  
  }

  // Return the Candidate with the smallest dR
  return match;

}

// Match Candidate T to a Candidate U in the Collection based on minimum Delta-Et
template <typename T, typename Collection>
const typename Collection::value_type *PFBenchmarkAlgo::matchByDeltaEt(const T *c1, const Collection *candidates) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of the Candidate and Collection
  if (!c1) throw cms::Exception("Invalid Arg") << "attempted to match invalid Candidate";
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to match to invalid Collection";

  double minDeltaEt = 9999.;
  const U *match = NULL;

  // Loop Over the Candidates...
  for (unsigned int i = 0; i < candidates->size(); i++) {

    const T *c2 = &(*candidates)[i];
    if (!c2) throw cms::Exception("Invalid Arg") << "attempted to match to invalid Candidate";

    // Find Minimal Delta-R
    double dEt = fabs(deltaEt(c1,c2));
    if (dEt <= minDeltaEt) { match = c2; minDeltaEt = dEt; }

  }

  // Return the Candidate with the smallest dR
  return match;

}

// Copy the Collection (useful when sorting)
template <typename T, typename Collection>
Collection PFBenchmarkAlgo::copyCollection(const Collection *candidates) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of the Collection
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to copy invalid Collection";

  Collection copy;

  for (unsigned int i = 0; i < candidates->size(); i++)
    vector_add(&(*candidates)[i],copy);

  return copy;

}


// Sort the U Candidates to the Candidate T based on minimum Delta-R
template <typename T, typename Collection>
void PFBenchmarkAlgo::sortByDeltaR(const T *c1, Collection &candidates) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of Candidate and Collection
  if (!c1) throw cms::Exception("Invalid Arg") << "attempted to sort by invalid Candidate";
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to sort invalid Candidates";

  // Sort the collection
  vector_sort(candidates,deltaRSorter<T,U>(c1));

}

// Sort the U Candidates to the Candidate T based on minimum Delta-Et
template <typename T, typename Collection>
void PFBenchmarkAlgo::sortByDeltaEt(const T *c1, Collection &candidates) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of Candidate and Collection
  if (!c1) throw cms::Exception("Invalid Arg") << "attempted to sort by invalid Candidate";
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to sort invalid Candidates";

  // Sort the collection
  vector_sort(candidates,deltaEtSorter<T,U>(c1));

}

// Constrain the U Candidates to the T Candidate based on Delta-R to T
template <typename T, typename Collection>
Collection PFBenchmarkAlgo::findAllInCone(const T *c1, const Collection *candidates, double ConeSize) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of Candidate and the Collection
  if (!c1) throw cms::Exception("Invalid Arg") << "attempted to constrain to invalid Candidate";
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to constrain invalid Collection";
  if (ConeSize <= 0) throw cms::Exception("Invalid Arg") << "zero or negative cone size specified";

  Collection constrained;

  for (unsigned int i = 0; i < candidates->size(); i++) {

    const U *c2 = &(*candidates)[i];

    // Add in-cone Candidates to the new Collection
    double dR = deltaR(c1,c2);
    if (dR < ConeSize) vector_add(c2,constrained);

  }

  // Sort and return Collection
  sortByDeltaR(c1,constrained);
  return constrained;

}

// Constrain the U Candidates to the T Candidate based on Delta-Et to T
template <typename T, typename Collection>
Collection PFBenchmarkAlgo::findAllInEtWindow(const T *c1, const Collection *candidates, double EtWindow) {

  typedef typename Collection::value_type U;

  // Try to verify the validity of Candidate and the Collection
  if (!c1) throw cms::Exception("Invalid Arg") << "attempted to constrain to invalid Candidate";
  if (!candidates) throw cms::Exception("Invalid Arg") << "attempted to constrain invalid Collection";
  if (EtWindow <= 0) throw cms::Exception("Invalid Arg") << "zero or negative cone size specified";

  Collection constrained;

  //CandidateCollection::const_iterator candidate;
  //for (candidate = candidates->begin(); candidate != candidates->end(); candidate++) {
  for (unsigned int i = 0; i < candidates->size(); i++) {

    const U *c2 = &(*candidates)[i];

    // Add in-cone Candidates to the new Collection
    double dEt = fabs(deltaEt(c1,c2));
    if (dEt < EtWindow) vector_add(c2,constrained);

  }

  // Sort and return Collection
  sortByDeltaEt(c1,constrained);
  return constrained;

}

#endif // RecoParticleFlow_Benchmark_PFBenchmarkAlgo_h
