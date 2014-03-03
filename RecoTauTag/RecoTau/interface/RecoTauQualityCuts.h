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
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace reco { namespace tau {

class RecoTauQualityCuts 
{
 public:
  // Quality cut types
  typedef boost::function<bool (const TrackBaseRef&)> TrackQCutFunc;  
  typedef std::vector<TrackQCutFunc> TrackQCutFuncCollection;
  typedef boost::function<bool (const PFCandidate&)> CandQCutFunc;  
  typedef std::vector<CandQCutFunc> CandQCutFuncCollection;
  typedef std::map<PFCandidate::ParticleType, CandQCutFuncCollection> CandQCutFuncMap;
  
  explicit RecoTauQualityCuts(const edm::ParameterSet& qcuts);

  /// Update the primary vertex
  void setPV(const reco::VertexRef& vtx) const { pv_ = vtx; }
    
  /// Update the leading track
  void setLeadTrack(const reco::TrackRef& leadTrack) const;
  void setLeadTrack(const reco::PFCandidate& leadCand) const;

  /// Update the leading track (using reference)
  /// If null, this will set the lead track ref null.
  void setLeadTrack(const reco::PFCandidateRef& leadCand) const;

  /// Filter a single Track
  bool filterTrack(const reco::TrackBaseRef& track) const;
  bool filterTrack(const reco::TrackRef& track) const;

  /// Filter a collection of Tracks
  template<typename Coll> 
  Coll filterTracks(const Coll& coll, bool invert = false) const 
  {
    Coll output;
    BOOST_FOREACH( const typename Coll::value_type track, coll ) {
      if ( filterTrack(track)^invert ) output.push_back(track);
    }
    return output;
  }

  /// Filter a single PFCandidate
  bool filterCand(const reco::PFCandidate& cand) const;

  /// Filter a PFCandidate held by a smart pointer or Ref
  template<typename PFCandRefType>
  bool filterCandRef(const PFCandRefType& cand) const { return filterCand(*cand); }

  /// Filter a ref vector of PFCandidates
  template<typename Coll> 
  Coll filterCandRefs(const Coll& refcoll, bool invert = false) const 
  {
    Coll output;
    BOOST_FOREACH( const typename Coll::value_type cand, refcoll ) {
      if ( filterCandRef(cand)^invert ) output.push_back(cand);
    }
    return output;
  }

 private:
  template <typename T> bool filterTrack_(const T& trackRef) const;
  bool filterGammaCand(const reco::PFCandidate& cand) const;
  bool filterNeutralHadronCand(const reco::PFCandidate& cand) const;
  bool filterCandByType(const reco::PFCandidate& cand) const;

  // The current primary vertex
  mutable reco::VertexRef pv_;
  // The current lead track references
  mutable reco::TrackBaseRef leadTrack_;

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

}} // end reco::tau:: namespace

#endif
