#ifndef RecoTauTag_RecoTau_ConeTools_h
#define RecoTauTag_RecoTau_ConeTools_h

#include "DataFormats/Math/interface/deltaR.h"
#include <boost/iterator/filter_iterator.hpp>
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include <functional>

namespace reco { namespace tau { namespace cone {

// Predicate class that tests if a candidate lies within some deltaR (min,
// max) about a supplied axis
template<class CandType>
class DeltaRFilter : public std::unary_function<CandType, bool> {
  public:
    DeltaRFilter(const reco::Candidate::LorentzVector& axis,
                 double min, double max): axis_(axis), min_(min), max_(max) {}
    bool operator()(const CandType& b) const {
      double deltaR = reco::deltaR<reco::Candidate::LorentzVector>
          (axis_, b.p4());
      return(deltaR >= min_ && deltaR < max_);
    }
  private:
    reco::Candidate::LorentzVector axis_;
    const double min_;
    const double max_;
};

// Wrapper around DeltaRFilter to support reference types like Ptr<>
template<class CandType>
class DeltaRPtrFilter : public std::unary_function<CandType, bool> {
  public:
    DeltaRPtrFilter(const reco::Candidate::LorentzVector& axis,
                    double min, double max): filter_(axis, min, max) {}
    bool operator()(const CandType& b) const { return filter_(*b); }
  private:
    DeltaRFilter<typename CandType::value_type> filter_;
};

/* Define our filters */
typedef DeltaRPtrFilter<PFCandidatePtr> PFCandPtrDRFilter;
typedef boost::filter_iterator< PFCandPtrDRFilter,
        std::vector<PFCandidatePtr>::const_iterator> PFCandPtrDRFilterIter;

typedef DeltaRFilter<PFRecoTauChargedHadron> ChargedHadronDRFilter;
typedef boost::filter_iterator< ChargedHadronDRFilter,
        std::vector<PFRecoTauChargedHadron>::const_iterator> ChargedHadronDRFilterIter;

typedef DeltaRFilter<RecoTauPiZero> PiZeroDRFilter;
typedef boost::filter_iterator< PiZeroDRFilter,
        std::vector<RecoTauPiZero>::const_iterator> PiZeroDRFilterIter;

}}}  // end namespace reco::tau::cone

#endif
