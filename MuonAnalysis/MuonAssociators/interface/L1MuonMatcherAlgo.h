#ifndef MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
#define MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
//
// $Id: L1MuonMatcherAlgo.h,v 1.9 2011/04/27 23:13:21 gpetrucc Exp $
//

/**
  \class    L1MuonMatcherAlgo L1MuonMatcherAlgo.h "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMatcherAlgo.h,v 1.9 2011/04/27 23:13:21 gpetrucc Exp $
*/


#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/AnySelector.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"

class L1MuonMatcherAlgo {
    public:
        explicit L1MuonMatcherAlgo(const edm::ParameterSet & iConfig) ;
        ~L1MuonMatcherAlgo() ;

        /// Call this method at the beginning of each run, to initialize geometry, magnetic field and propagators
        void init(const edm::EventSetup &iSetup) ;

        /// Extrapolate reco::Track to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const reco::Track &tk)     const { return prop_.extrapolate(tk); }

        /// Extrapolate reco::Candidate to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const reco::Candidate &tk) const { return prop_.extrapolate(tk); }

        /// Extrapolate a SimTrack to the muon station 2, return an invalid TSOS if it fails. Requires SimVertices to know where to start from.
        TrajectoryStateOnSurface extrapolate(const SimTrack &tk, const edm::SimVertexContainer &vtx) const { return prop_.extrapolate(tk, vtx); }

        /// Extrapolate a FreeTrajectoryState to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState &state) const { return prop_.extrapolate(state); }

        /// Return the propagator to second muon station (in case it's needed)
        PropagateToMuon & propagatorToMuon() { return prop_; }
        /// Return the propagator to second muon station (in case it's needed)
        const PropagateToMuon & propagatorToMuon() const { return prop_; }

        /// Try to match one track to one L1. Return true if succeeded (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// The preselection cut on L1, if specified in the config, is applied before the match
        bool match(const reco::Track &tk, const l1extra::L1MuonParticle &l1, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(tk);
            return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : false;
        }

        /// Try to match one track to one L1. Return true if succeeded (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// The preselection cut on L1, if specified in the config, is applied before the match
        bool match(const reco::Candidate &c, const l1extra::L1MuonParticle &l1, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(c);
            return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : false;
        }

        /// Try to match one simtrack to one L1. Return true if succeeded (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// The preselection cut on L1, if specified in the config, is applied before the match
        bool match(const SimTrack &tk, const edm::SimVertexContainer &vtxs, const l1extra::L1MuonParticle &l1, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(tk, vtxs);
            return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : false;
        }


        /// Try to match one track to one L1. Return true if succeeded (and update deltaR, deltaPhi accordingly)
        /// The preselection cut on L1, if specified in the config, is applied before the match
        bool match(TrajectoryStateOnSurface & propagated, const l1extra::L1MuonParticle &l1, float &deltaR, float &deltaPhi) const ;

        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// Returns -1 if the match fails
        /// The preselection cut on L1, if specified in the config, is applied before the match
        int match(const reco::Track &tk, const std::vector<l1extra::L1MuonParticle> &l1, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(tk);
            return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
        }

        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// Returns -1 if the match fails
        /// The preselection cut on L1, if specified in the config, is applied before the match
        int match(const reco::Candidate &c, const std::vector<l1extra::L1MuonParticle> &l1, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(c);
            return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
        }

        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// Returns -1 if the match fails
        /// The preselection cut on L1, if specified in the config, is applied before the match
        int match(const SimTrack &tk, const edm::SimVertexContainer &vtxs, const std::vector<l1extra::L1MuonParticle> &l1, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(tk, vtxs);
            return propagated.isValid() ? match(propagated, l1, deltaR, deltaPhi) : -1;
        }


        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi accordingly)
        /// Returns -1 if the match fails
        /// The preselection cut on L1, if specified in the config, is applied before the match
        int match(TrajectoryStateOnSurface &propagated, const std::vector<l1extra::L1MuonParticle> &l1, float &deltaR, float &deltaPhi) const ;


        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// Returns -1 if the match fails
        /// Only the objects passing the selector will be allowed for the match.
        /// If you don't need a selector, just use an AnySelector (CommonTools/Utils) which accepts everything
        template<typename Collection, typename Selector>
        int matchGeneric(const reco::Track &tk, const Collection &l1, const Selector &sel, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(tk);
            return propagated.isValid() ? matchGeneric(propagated, l1, sel, deltaR, deltaPhi) : -1;
        }

        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi and propagated TSOS accordingly)
        /// Returns -1 if the match fails
        /// Only the objects passing the selector will be allowed for the match.
        /// If you don't need a selector, just use an AnySelector (CommonTools/Utils) which accepts everything
        template<typename Collection, typename Selector>
        int matchGeneric(const reco::Candidate &c, const Collection &l1, const Selector &sel, float &deltaR, float &deltaPhi, TrajectoryStateOnSurface &propagated) const {
            propagated = extrapolate(c);
            return propagated.isValid() ? matchGeneric(propagated, l1, sel, deltaR, deltaPhi) : -1;
        }

        /// Find the best match to L1, and return its index in the vector (and update deltaR, deltaPhi accordingly)
        /// Returns -1 if the match fails
        /// Only the objects passing the selector will be allowed for the match.
        /// The selector defaults to an AnySelector (CommonTools/Utils) which just accepts everything
        template<typename Collection, typename Selector>
        int matchGeneric(TrajectoryStateOnSurface &propagated, const Collection &l1, const Selector &sel, float &deltaR, float &deltaPhi) const ;


        /// Add this offset to the L1 phi before doing the match, to correct for different scales in L1 vs offline
        void setL1PhiOffset(double l1PhiOffset) { l1PhiOffset_ = l1PhiOffset; }

    private:
        PropagateToMuon prop_;

        typedef StringCutObjectSelector<l1extra::L1MuonParticle> L1Selector;
        /// Preselection cut to apply to L1 candidates before matching
        L1Selector preselectionCut_;

        /// Matching cuts
        double deltaR2_, deltaPhi_, deltaEta_;

        /// Sort by deltaPhi or deltaEta instead of deltaR
        enum SortBy { SortByDeltaR, SortByDeltaPhi, SortByDeltaEta, SortByPt };
        SortBy sortBy_;

        /// offset to be added to the L1 phi before the match
        double l1PhiOffset_;
};

template<typename Collection, typename Selector>
int 
L1MuonMatcherAlgo::matchGeneric(TrajectoryStateOnSurface &propagated, const Collection &l1s, const Selector &sel, float &deltaR, float &deltaPhi) const {
    typedef typename Collection::value_type obj;
    int match = -1;
    double minDeltaPhi = deltaPhi_;
    double minDeltaEta = deltaEta_;
    double minDeltaR2  = deltaR2_;
    double minPt       = -1.0;
    GlobalPoint pos = propagated.globalPosition();
    for (int i = 0, n = l1s.size(); i < n; ++i) {
        const obj &l1 = l1s[i];
        if (sel(l1)) {
            double thisDeltaPhi = ::deltaPhi(double(pos.phi()),  l1.phi()+l1PhiOffset_);
            double thisDeltaEta = pos.eta() - l1.eta();
            double thisDeltaR2  = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi()+l1PhiOffset_);
            double thisPt       = l1.pt();
            if ((fabs(thisDeltaPhi) < deltaPhi_) && (fabs(thisDeltaEta) < deltaEta_) && (thisDeltaR2 < deltaR2_)) { // check all
                bool betterMatch = (match == -1);
                switch (sortBy_) {
                    case SortByDeltaR:   betterMatch = (thisDeltaR2        < minDeltaR2);        break;
                    case SortByDeltaEta: betterMatch = (fabs(thisDeltaEta) < fabs(minDeltaEta)); break;
                    case SortByDeltaPhi: betterMatch = (fabs(thisDeltaPhi) < fabs(minDeltaPhi)); break;
                    case SortByPt:       betterMatch = (thisPt             > minPt);             break;
                }
                if (betterMatch) { // sort on one
                    match = i;
                    deltaR   = std::sqrt(thisDeltaR2);
                    deltaPhi = thisDeltaPhi;
                    minDeltaR2  = thisDeltaR2;
                    minDeltaEta = thisDeltaEta;
                    minDeltaPhi = thisDeltaPhi;
                    minPt       = thisPt;
                }
            }
        }
    }
    return match;
}

#endif
