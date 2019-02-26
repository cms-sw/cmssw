#ifndef RecoTauTag_RecoTau_RecoTauCrossCleaning_h
#define RecoTauTag_RecoTau_RecoTauCrossCleaning_h

#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"

namespace reco::tau::xclean {

/// Transform a pizero to remove given candidates
template<typename PtrIter>
class CrossCleanPiZeros 
{
 public:
  typedef std::vector<RecoTauPiZero> PiZeroList;

  CrossCleanPiZeros(const PtrIter& chargedHadronsBegin, const PtrIter& chargedHadronsEnd, int mode = kRemoveChargedAndNeutralDaughterOverlaps) 
    : mode_(mode)
  {
    initialize(chargedHadronsBegin, chargedHadronsEnd);
  }

  void initialize(const PtrIter& chargedHadronsBegin, const PtrIter& chargedHadronsEnd)
  {
    // CV: make sure this never gets called.
    assert(0);
  }

  /// Return a vector of pointers to pizeros.  PiZeros that needed cleaning
  /// are cloned, modified, and owned by this class.  The un-modified pointers
  /// point to objects in the [input] vector.
  PiZeroList operator()(const std::vector<RecoTauPiZero>& input) const 
  {
    PiZeroList output;
    output.reserve(input.size());
    for(auto const& piZero : input ) {
      const RecoTauPiZero::daughters& daughters = piZero.daughterPtrVector();
      std::set<reco::CandidatePtr> toCheck(daughters.begin(), daughters.end());
      std::vector<reco::CandidatePtr> cleanDaughters;
      std::set_difference(toCheck.begin(), toCheck.end(), toRemove_.begin(), toRemove_.end(), std::back_inserter(cleanDaughters));
      if ( cleanDaughters.size() == daughters.size() ) {
	// We don't need to clean anything, just add a pointer to current pizero
	output.push_back(piZero);
      } else {
	// Otherwise rebuild
	RecoTauPiZero newPiZero = piZero;
	newPiZero.clearDaughters();
	// Add our cleaned daughters.
	for(auto const& ptr : cleanDaughters ) {
	  newPiZero.addDaughter(ptr);
	}
	// Check if the pizero is not empty.  If empty, forget it.
	if ( newPiZero.numberOfDaughters() ) {
	  p4Builder_.set(newPiZero);
	  // Make our ptr container take ownership.
	  output.push_back(newPiZero);
	}
      }
    }
    return output;
  }

  enum { kRemoveChargedDaughterOverlaps, kRemoveChargedAndNeutralDaughterOverlaps };
 
 private:
  int mode_;
  AddFourMomenta p4Builder_;
  std::set<reco::CandidatePtr> toRemove_;
};

// Determine if a candidate is contained in a collection of charged hadrons or pizeros.
template<typename PtrIter>
class CrossCleanPtrs 
{
 public:
  CrossCleanPtrs(const PtrIter& particlesBegin, const PtrIter& particlesEnd)
  {
    initialize(particlesBegin, particlesEnd);
  }

  void initialize(const PtrIter& particlesBegin, const PtrIter& particlesEnd)
  {
    // CV: make sure this never gets called.
    assert(0);
  }

  template<typename AnyPtr>
  bool operator() (const AnyPtr& ptr) const 
  {
    if ( toRemove_.count(CandidatePtr(ptr)) ) return false;
    else return true;
  }
 private:
  std::set<CandidatePtr> toRemove_;
};

// Predicate to filter PFCandPtrs (and those compatible to this type) by the
// particle id
class FilterPFCandByParticleId {
  public:
    FilterPFCandByParticleId(int particleId):
      id_(particleId){};
    template<typename PFCandCompatiblePtrType>
      bool operator()(const PFCandCompatiblePtrType& ptr) const {
        return ptr->particleId() == id_;
      }
  private:
    int id_;
};

// Create the AND of two predicates
template<typename P1, typename P2>
class PredicateAND {
  public:
    PredicateAND(const P1& p1, const P2& p2):
      p1_(p1),p2_(p2){}

    template<typename AnyPtr>
    bool operator() (const AnyPtr& ptr) const {
      return (p1_(ptr) && p2_(ptr));
    }
  private:
    const P1& p1_;
    const P2& p2_;
};

// Helper function to infer template type
template<typename P1, typename P2>
PredicateAND<P1, P2> makePredicateAND(const P1& p1, const P2& p2) {
  return PredicateAND<P1, P2>(p1, p2);
}

}

#endif
