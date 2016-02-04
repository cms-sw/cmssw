#ifndef MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
#define MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
//
// $Id: L1MuonMatcherAlgo.h,v 1.1 2010/04/15 18:37:17 klukas Exp $
//

/**
  \class    L1MuonMatcherAlgo L1MuonMatcherAlgo.h "HLTriggerOffline/Muon/interface/L1MuonMatcherAlgo.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMatcherAlgo.h,v 1.1 2010/04/15 18:37:17 klukas Exp $
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
#include "HLTriggerOffline/Muon/interface/PropagateToMuon.h"

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


    private:
        PropagateToMuon prop_;

        typedef StringCutObjectSelector<l1extra::L1MuonParticle> L1Selector;
        /// Preselection cut to apply to L1 candidates before matching
        L1Selector preselectionCut_;

        /// Matching cuts
        double deltaR2_, deltaPhi_;

        /// Sort by deltaPhi instead of deltaR
        bool sortByDeltaPhi_;
};

template<typename Collection, typename Selector>
int 
L1MuonMatcherAlgo::matchGeneric(TrajectoryStateOnSurface &propagated, const Collection &l1s, const Selector &sel, float &deltaR, float &deltaPhi) const {
    typedef typename Collection::value_type obj;
    int match = -1;
    double minDeltaPhi = deltaPhi_;
    double minDeltaR2  = deltaR2_;
    GlobalPoint pos = propagated.globalPosition();
    for (int i = 0, n = l1s.size(); i < n; ++i) {
        const obj &l1 = l1s[i];
        if (sel(l1)) {
            double thisDeltaPhi = ::deltaPhi(double(pos.phi()),  l1.phi());
            double thisDeltaR2  = ::deltaR2(double(pos.eta()), double(pos.phi()), l1.eta(), l1.phi());
            if ((fabs(thisDeltaPhi) < deltaPhi_) && (thisDeltaR2 < deltaR2_)) { // check both
                if (sortByDeltaPhi_ ? (fabs(thisDeltaPhi) < fabs(minDeltaPhi)) : (thisDeltaR2 < minDeltaR2)) { // sort on one
                    match = i;
                    deltaR   = std::sqrt(thisDeltaR2);
                    deltaPhi = thisDeltaPhi;
                    if (sortByDeltaPhi_) minDeltaPhi = thisDeltaPhi; else minDeltaR2 = thisDeltaR2;
                }
            }
        }
    }
    return match;
}

#endif
