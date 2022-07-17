#ifndef MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
#define MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
/**
  \class    L1MuonMatcherAlgo L1MuonMatcherAlgo.h "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMatcherAlgo.h,v 1.8 2010/07/01 07:40:52 gpetrucc Exp $
*/
#include <cmath>
#include <type_traits>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/AnySelector.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"

template <edm::Transition Tr>
class L1MuonMatcherAlgoT {
public:
  explicit L1MuonMatcherAlgoT(const edm::ParameterSet &iConfig, edm::ConsumesCollector);
  ~L1MuonMatcherAlgoT() = default;

  /// Call this method at the beginning of each run, to initialize geometry, magnetic field and propagators
  void init(const edm::EventSetup &iSetup);

  /// Extrapolate reco::Track to the muon station 2, return an invalid TSOS if it fails
  TrajectoryStateOnSurface extrapolate(const reco::Track &tk) const { return prop_.extrapolate(tk); }

  /// Extrapolate reco::Candidate to the muon station 2, return an invalid TSOS if it fails
  TrajectoryStateOnSurface extrapolate(const reco::Candidate &tk) const { return prop_.extrapolate(tk); }

  /// Extrapolate a SimTrack to the muon station 2, return an invalid TSOS if it fails. Requires SimVertices to know where to start from.
  TrajectoryStateOnSurface extrapolate(const SimTrack &tk, const edm::SimVertexContainer &vtx) const {
    return prop_.extrapolate(tk, vtx);
  }

  /// Extrapolate a FreeTrajectoryState to the muon station 2, return an invalid TSOS if it fails
  TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState &state) const { return prop_.extrapolate(state); }

  /// Return the propagator to second muon station (in case it's needed)
  PropagateToMuon &propagatorToMuon() { return prop_; }
  /// Return the propagator to second muon station (in case it's needed)
  const PropagateToMuon &propagatorToMuon() const { return prop_; }

  /// Try to match one track to one L1. Return true if succeeded (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// The preselection cut on L1, if specified in the config, is applied before the match
  bool match(const reco::Track &tk,
             const l1extra::L1MuonParticle &l1,
             float &deltaR,
             float &deltaPhi,
             TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : false;
  }

  /// Try to match one track to one L1. Return true if succeeded (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// The preselection cut on L1, if specified in the config, is applied before the match
  bool match(const reco::Candidate &c,
             const l1extra::L1MuonParticle &l1,
             float &deltaR,
             float &deltaPhi,
             TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(c);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : false;
  }

  /// Try to match one simtrack to one L1. Return true if succeeded (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// The preselection cut on L1, if specified in the config, is applied before the match
  bool match(const SimTrack &tk,
             const edm::SimVertexContainer &vtxs,
             const l1extra::L1MuonParticle &l1,
             float &deltaR,
             float &deltaPhi,
             TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk, vtxs);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : false;
  }

  /// Try to match one track to one L1. Return true if succeeded (and update deltaR, deltaPhi accordingly)
  /// The preselection cut on L1, if specified in the config, is applied before the match
  bool match(TrajectoryStateOnSurface &propagated,
             const l1extra::L1MuonParticle &l1,
             float &deltaR,
             float &deltaPhi) const;

  // Methods to match with vectors of legacy L1 muon trigger objects

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on L1, if specified in the config, is applied before the match
  int match(const reco::Track &tk,
            const std::vector<l1extra::L1MuonParticle> &l1,
            float &deltaR,
            float &deltaPhi,
            TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on L1, if specified in the config, is applied before the match
  int match(const reco::Candidate &c,
            const std::vector<l1extra::L1MuonParticle> &l1,
            float &deltaR,
            float &deltaPhi,
            TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(c);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on L1, if specified in the config, is applied before the match
  int match(const SimTrack &tk,
            const edm::SimVertexContainer &vtxs,
            const std::vector<l1extra::L1MuonParticle> &l1,
            float &deltaR,
            float &deltaPhi,
            TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk, vtxs);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on L1, if specified in the config, is applied before the match
  int match(TrajectoryStateOnSurface &propagated,
            const std::vector<l1extra::L1MuonParticle> &l1,
            float &deltaR,
            float &deltaPhi) const;

  // Methods to match with vectors of stage2 L1 muon trigger objects

  /// Find the best match to stage2 L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on stage2 L1, if specified in the config, is applied before the match
  int match(const reco::Track &tk,
            const std::vector<l1t::Muon> &l1,
            float &deltaR,
            float &deltaPhi,
            TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to stage2 L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on stage2 L1, if specified in the config, is applied before the match
  int match(const reco::Candidate &c,
            const std::vector<l1t::Muon> &l1,
            float &deltaR,
            float &deltaPhi,
            TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(c);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to stage2 L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on stage 2 L1, if specified in the config, is applied before the match
  int match(const SimTrack &tk,
            const edm::SimVertexContainer &vtxs,
            const std::vector<l1t::Muon> &l1,
            float &deltaR,
            float &deltaPhi,
            TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk, vtxs);
    return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to stage 2 L1, and return its index in the vector (and update deltaR, deltaPhi accordingly)
  /// Returns -1 if the match fails
  /// The preselection cut on stage 2 L1, if specified in the config, is applied before the match
  int match(TrajectoryStateOnSurface &propagated,
            const std::vector<l1t::Muon> &l1,
            float &deltaR,
            float &deltaPhi) const;

  // Generic matching

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// Only the objects passing the selector will be allowed for the match.
  /// If you don't need a selector, just use an AnySelector (CommonTools/Utils) which accepts everything
  template <typename Collection, typename Selector>
  int matchGeneric(const reco::Track &tk,
                   const Collection &l1,
                   const Selector &sel,
                   float &deltaR,
                   float &deltaPhi,
                   TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(tk);
    return propagated.isValid() ? matchGeneric(propagated, l1, sel, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
  /// Returns -1 if the match fails
  /// Only the objects passing the selector will be allowed for the match.
  /// If you don't need a selector, just use an AnySelector (CommonTools/Utils) which accepts everything
  template <typename Collection, typename Selector>
  int matchGeneric(const reco::Candidate &c,
                   const Collection &l1,
                   const Selector &sel,
                   float &deltaR,
                   float &deltaPhi,
                   TrajectoryStateOnSurface &propagated) const {
    propagated = extrapolate(c);
    return propagated.isValid() ? matchGeneric(propagated, l1, sel, deltaR, deltaPhi) : -1;
  }

  /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi accordingly)
  /// Returns -1 if the match fails
  /// Only the objects passing the selector will be allowed for the match.
  /// The selector defaults to an AnySelector (CommonTools/Utils) which just accepts everything
  template <typename Collection, typename Selector>
  int matchGeneric(TrajectoryStateOnSurface &propagated,
                   const Collection &l1,
                   const Selector &sel,
                   float &deltaR,
                   float &deltaPhi) const;

  /// Add this offset to the L1 phi before doing the match, to correct for different scales in L1 vs offline
  void setL1PhiOffset(double l1PhiOffset) { l1PhiOffset_ = l1PhiOffset; }

private:
  template <class T>
  int genericQuality(T const &cand) const {
    return 0;
  }

  int genericQuality(l1extra::L1MuonParticle const &cand) const { return cand.gmtMuonCand().rank(); }
  int genericQuality(l1t::Muon const &cand) const { return cand.hwQual(); }

  PropagateToMuonSetupT<Tr> propSetup_;
  PropagateToMuon prop_;

  bool useStage2L1_;

  typedef StringCutObjectSelector<reco::Candidate, true> L1Selector;
  /// Preselection cut to apply to L1 candidates before matching
  L1Selector preselectionCut_;

  /// Matching cuts
  double deltaR2_, deltaPhi_, deltaEta_;

  /// Sort by deltaPhi or deltaEta instead of deltaR
  enum SortBy { SortByDeltaR = 0, SortByDeltaPhi, SortByDeltaEta, SortByPt, SortByQual };
  SortBy sortBy_;

  /// offset to be added to the L1 phi before the match
  double l1PhiOffset_;
};

template <edm::Transition Tr>
template <typename Collection, typename Selector>
int L1MuonMatcherAlgoT<Tr>::matchGeneric(TrajectoryStateOnSurface &propagated,
                                         const Collection &l1s,
                                         const Selector &sel,
                                         float &deltaR,
                                         float &deltaPhi) const {
  typedef typename Collection::value_type obj;

  int match = -1;
  double minDeltaPhi = deltaPhi_;
  double minDeltaEta = deltaEta_;
  double minDeltaR2 = deltaR2_;
  double maxPt = -1.0;
  int maxQual = 0;
  GlobalPoint pos = propagated.globalPosition();
  for (int i = 0, n = l1s.size(); i < n; ++i) {
    const obj &l1 = l1s[i];
    if (sel(l1)) {
      double thisDeltaPhi = ::deltaPhi(double(pos.phi()), l1.phi() + l1PhiOffset_);
      double thisDeltaEta = pos.eta() - l1.eta();
      double thisDeltaR2 = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi() + l1PhiOffset_);
      int thisQual = genericQuality<obj>(l1);
      double thisPt = l1.pt();
      if ((fabs(thisDeltaPhi) < deltaPhi_) && (fabs(thisDeltaEta) < deltaEta_) &&
          (thisDeltaR2 < deltaR2_)) {  // check all
        bool betterMatch = (match == -1);
        switch (sortBy_) {
          case SortByDeltaR:
            betterMatch = (thisDeltaR2 < minDeltaR2);
            break;
          case SortByDeltaEta:
            betterMatch = (fabs(thisDeltaEta) < fabs(minDeltaEta));
            break;
          case SortByDeltaPhi:
            betterMatch = (fabs(thisDeltaPhi) < fabs(minDeltaPhi));
            break;
          case SortByPt:
            betterMatch = (thisPt > maxPt);
            break;
          // Quality is an int, adding sorting by pT in case of identical qualities
          case SortByQual:
            betterMatch = (thisQual > maxQual || ((thisQual == maxQual) && (thisPt > maxPt)));
            break;
        }
        if (betterMatch) {  // sort on one
          match = i;
          deltaR = std::sqrt(thisDeltaR2);
          deltaPhi = thisDeltaPhi;
          minDeltaR2 = thisDeltaR2;
          minDeltaEta = thisDeltaEta;
          minDeltaPhi = thisDeltaPhi;
          maxQual = thisQual;
          maxPt = thisPt;
        }
      }
    }
  }
  return match;
}

template <edm::Transition Tr>
L1MuonMatcherAlgoT<Tr>::L1MuonMatcherAlgoT(const edm::ParameterSet &iConfig, edm::ConsumesCollector iCollector)
    : propSetup_(iConfig, iCollector),
      useStage2L1_(iConfig.existsAs<bool>("useStage2L1") ? iConfig.getParameter<bool>("useStage2L1") : false),
      preselectionCut_(
          (iConfig.existsAs<std::string>("preselection")) ? iConfig.getParameter<std::string>("preselection") : ""),
      deltaR2_(std::pow(iConfig.getParameter<double>("maxDeltaR"), 2)),
      deltaPhi_(iConfig.existsAs<double>("maxDeltaPhi") ? iConfig.getParameter<double>("maxDeltaPhi") : 10),
      deltaEta_(iConfig.existsAs<double>("maxDeltaEta") ? iConfig.getParameter<double>("maxDeltaEta") : 10),
      l1PhiOffset_(iConfig.existsAs<double>("l1PhiOffset") ? iConfig.getParameter<double>("l1PhiOffset") : 0) {
  bool reqQual = iConfig.existsAs<bool>("sortByQual") && iConfig.getParameter<bool>("sortByQual");
  bool reqPhi = iConfig.existsAs<bool>("sortByDeltaPhi") && iConfig.getParameter<bool>("sortByDeltaPhi");
  bool reqEta = iConfig.existsAs<bool>("sortByDeltaEta") && iConfig.getParameter<bool>("sortByDeltaEta");
  bool reqPt = iConfig.existsAs<bool>("sortByPt") && iConfig.getParameter<bool>("sortByPt");
  std::string sortBy = iConfig.existsAs<std::string>("sortBy") ? iConfig.getParameter<std::string>("sortBy") : "";
  if (reqPhi + reqEta + reqPt + reqQual > 1)
    throw cms::Exception("Configuration")
        << "L1MuonMatcherAlgoT: Can't set more than one 'sortBy<XXX>' parameter to True.\n";
  if (sortBy == "deltaPhi") {
    if (reqEta || reqPt || reqQual)
      throw cms::Exception("Configuration") << "L1MuonMatcherAlgoT: Can't set sortBy = 'deltaPhi' and set also another "
                                               "'sortBy<XXX>' parameter to True.\n";
    else
      reqPhi = true;
  }
  if (sortBy == "deltaEta") {
    if (reqPhi || reqPt || reqQual)
      throw cms::Exception("Configuration") << "L1MuonMatcherAlgoT: Can't set sortBy = 'deltaEta' and set also another "
                                               "'sortBy<XXX>' parameter to True.\n";
    else
      reqEta = true;
  }
  if (sortBy == "pt") {
    if (reqPhi || reqEta || reqQual)
      throw cms::Exception("Configuration")
          << "L1MuonMatcherAlgoT: Can't set sortBy = 'pt' and set also another 'sortBy<XXX>' parameter to True.\n";
    else
      reqPt = true;
  }
  if (sortBy == "quality") {
    if (reqPhi || reqEta || reqPt)
      throw cms::Exception("Configuration")
          << "L1MuonMatcherAlgoT: Can't set sortBy = 'quality' and set also another 'sortBy<XXX>' parameter to True.\n";
    else
      reqQual = true;
  }
  if (sortBy == "deltaR") {
    if (reqPhi || reqEta || reqPt || reqQual)
      throw cms::Exception("Configuration")
          << "L1MuonMatcherAlgoT: Can't set sortBy = 'deltaR' and set also another 'sortBy<XXX>' parameter to True.\n";
  }
  // so, if we're here there's no ambiguity in what the user may want. either everything is false, or exactly one req is true.
  if (reqEta)
    sortBy_ = SortByDeltaEta;
  else if (reqPhi)
    sortBy_ = SortByDeltaPhi;
  else if (reqQual)
    sortBy_ = SortByQual;
  else if (reqPt)
    sortBy_ = SortByPt;
  else
    sortBy_ = SortByDeltaR;
}

template <edm::Transition Tr>
void L1MuonMatcherAlgoT<Tr>::init(const edm::EventSetup &iSetup) {
  prop_ = propSetup_.init(iSetup);
}

template <edm::Transition Tr>
bool L1MuonMatcherAlgoT<Tr>::match(TrajectoryStateOnSurface &propagated,
                                   const l1extra::L1MuonParticle &l1,
                                   float &deltaR,
                                   float &deltaPhi) const {
  if (preselectionCut_(l1)) {
    GlobalPoint pos = propagated.globalPosition();
    double thisDeltaPhi = ::deltaPhi(double(pos.phi()), l1.phi() + l1PhiOffset_);
    double thisDeltaEta = pos.eta() - l1.eta();
    double thisDeltaR2 = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi() + l1PhiOffset_);
    if ((fabs(thisDeltaPhi) < deltaPhi_) && (fabs(thisDeltaEta) < deltaEta_) && (thisDeltaR2 < deltaR2_)) {
      deltaR = std::sqrt(thisDeltaR2);
      deltaPhi = thisDeltaPhi;
      return true;
    }
  }
  return false;
}

template <edm::Transition Tr>
int L1MuonMatcherAlgoT<Tr>::match(TrajectoryStateOnSurface &propagated,
                                  const std::vector<l1extra::L1MuonParticle> &l1s,
                                  float &deltaR,
                                  float &deltaPhi) const {
  return matchGeneric(propagated, l1s, preselectionCut_, deltaR, deltaPhi);
}

template <edm::Transition Tr>
int L1MuonMatcherAlgoT<Tr>::match(TrajectoryStateOnSurface &propagated,
                                  const std::vector<l1t::Muon> &l1s,
                                  float &deltaR,
                                  float &deltaPhi) const {
  return matchGeneric(propagated, l1s, preselectionCut_, deltaR, deltaPhi);
}

using L1MuonMatcherAlgo = L1MuonMatcherAlgoT<edm::Transition::Event>;

#endif
