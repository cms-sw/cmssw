#ifndef RecoTauTag_RecoTau_RecoTauQualityCuts_h
#define RecoTauTag_RecoTau_RecoTauQualityCuts_h

/*
 * RecoTauQualityCuts
 *
 * Author: Evan K. Friis
 *
 * Constructs a number of independent requirements on Candidates by building
 * binary predicate functions.  These are held in a number of lists of
 * functions.  Each of these lists is mapped to a Candidate particle type
 * (like hadron, gamma, etc).  When a Candidate is passed to filter(),
 * the correct list is looked up, and the result is the AND of all the predicate
 * functions.  See the .cc files for the QCut functions.
 *
 * Note that for some QCuts, the primary vertex must be updated every event.
 * Others require the lead track be defined for each tau before filter(..)
 * is called.
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <functional>

namespace reco {
  namespace tau {

    class RecoTauQualityCuts {
    public:
      // Quality cut types
      typedef std::function<bool(const TrackBaseRef&)> TrackQCutFunc;
      typedef std::vector<TrackQCutFunc> TrackQCutFuncCollection;
      typedef std::function<bool(const Candidate&)> CandQCutFunc;
      typedef std::vector<CandQCutFunc> CandQCutFuncCollection;
      typedef std::map<int, CandQCutFuncCollection> CandQCutFuncMap;

      explicit RecoTauQualityCuts(const edm::ParameterSet& qcuts);

      /// Update the primary vertex
      void setPV(const reco::VertexRef& vtx) { pv_ = vtx; }

      /// Update the leading track
      void setLeadTrack(const reco::Track& leadTrack);
      void setLeadTrack(const reco::Candidate& leadCand);

      /// Update the leading track (using reference)
      /// If null, this will set the lead track ref null.
      void setLeadTrack(const reco::CandidateRef& leadCand);

      /// Filter a single Track
      bool filterTrack(const reco::TrackBaseRef& track) const;
      bool filterTrack(const reco::TrackRef& track) const;
      bool filterTrack(const reco::Track& track) const;
      /// or a single charged candidate
      bool filterChargedCand(const reco::Candidate& cand) const;

      /// Filter a collection of Tracks
      template <typename Coll>
      Coll filterTracks(const Coll& coll, bool invert = false) const {
        Coll output;
        for (auto const& track : coll) {
          if (filterTrack(track) ^ invert)
            output.push_back(track);
        }
        return output;
      }

      /// Filter a single Candidate
      bool filterCand(const reco::Candidate& cand) const;

      /// Filter a Candidate held by a smart pointer or Ref
      template <typename CandRefType>
      bool filterCandRef(const CandRefType& cand) const {
        return filterCand(*cand);
      }

      /// Filter a ref vector of Candidates
      template <typename Coll>
      Coll filterCandRefs(const Coll& refcoll, bool invert = false) const {
        Coll output;
        for (auto const& cand : refcoll) {
          if (filterCandRef(cand) ^ invert)
            output.push_back(cand);
        }
        return output;
      }

      /// Declare all parameters read from python config file
      static void fillDescriptions(edm::ParameterSetDescription& descriptions);

    private:
      bool filterTrack_(const reco::Track* track) const;
      bool filterGammaCand(const reco::Candidate& cand) const;
      bool filterNeutralHadronCand(const reco::Candidate& cand) const;
      bool filterCandByType(const reco::Candidate& cand) const;

      // The current primary vertex
      reco::VertexRef pv_;
      // The current lead track references
      const reco::Track* leadTrack_;

      double minTrackPt_;
      double maxTrackChi2_;
      int minTrackPixelHits_;
      int minTrackHits_;
      double maxTransverseImpactParameter_;
      double maxDeltaZ_;
      double maxDeltaZToLeadTrack_;
      double minTrackVertexWeight_;
      double minGammaEt_;
      double minNeutralHadronEt_;
      bool checkHitPattern_;
      bool checkPV_;
    };

    // Split an input set of quality cuts into those that need to be inverted
    // to select PU (the first member) and those that are general quality cuts.
    std::pair<edm::ParameterSet, edm::ParameterSet> factorizePUQCuts(const edm::ParameterSet& inputSet);

  }  // namespace tau
}  // namespace reco

#endif
