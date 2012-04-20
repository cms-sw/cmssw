#ifndef RecoTauTag_RecoTau_RecoTauQualityCuts_h
#define RecoTauTag_RecoTau_RecoTauQualityCuts_h

/*
 * RecoTauQualityCuts
 *
 * Author: Evan K. Friis
 *
 * Constructs a number of independent requirements on PFCandidates by building
 * binary predicate functions.  These are held in a number of lists of
 * functions.  Each of these lists is mapped to a PFCandidate particle type
 * (like hadron, gamma, etc).  When a PFCandidate is passed to filter(),
 * the correct list is looked up, and the result is the AND of all the predicate
 * functions.  See the .cc files for the QCut functions.
 *
 * Note that for some QCuts, the primary vertex must be updated every event.
 * Others require the lead track be defined for each tau before filter(..)
 * is called.
 *
 */

#include <boost/function.hpp>
#include <boost/foreach.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco { namespace tau {

class RecoTauQualityCuts {
  public:
    // Quality cut types
    typedef boost::function<bool (const PFCandidate&)> QCutFunc;
    typedef std::vector<QCutFunc> QCutFuncCollection;
    typedef std::map<PFCandidate::ParticleType, QCutFuncCollection> QCutFuncMap;

    explicit RecoTauQualityCuts(const edm::ParameterSet &qcuts);

    /// Update the primary vertex
    void setPV(const reco::VertexRef& vtx) const { pv_ = vtx; }

    /// Update the leading track
    void setLeadTrack(const reco::PFCandidate& leadCand) const;

    /// Get the predicate used to filter.
    const QCutFunc& predicate() const { return predicate_; }

    /// Filter a single PFCandidate
    bool filter(const reco::PFCandidate& cand) const {
      return predicate_(cand);
    }

    /// Filter a PFCandidate held by a smart pointer or Ref
    template<typename PFCandRefType>
    bool filterRef(const PFCandRefType& cand) const { return filter(*cand); }

    /// Filter a ref vector of PFCandidates
    template<typename Coll> Coll filterRefs(
        const Coll& refcoll, bool invert=false) const {
      Coll output;
      BOOST_FOREACH(const typename Coll::value_type cand, refcoll) {
        if (filterRef(cand)^invert)
          output.push_back(cand);
      }
      return output;
    }

  private:
    // The current primary vertex
    mutable reco::VertexRef pv_;
    // The current lead track references
    mutable reco::TrackBaseRef leadTrack_;
    // A mapping from particle type to a set of QCuts
    QCutFuncMap qcuts_;
    // Our entire predicate function
    QCutFunc predicate_;
};

// Split an input set of quality cuts into those that need to be inverted
// to select PU (the first member) and those that are general quality cuts.
std::pair<edm::ParameterSet, edm::ParameterSet> factorizePUQCuts(
    const edm::ParameterSet& inputSet);

}}  // end reco::tau:: namespace
#endif
