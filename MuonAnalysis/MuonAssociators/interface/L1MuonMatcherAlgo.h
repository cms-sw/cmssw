#ifndef MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
#define MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
//
// $Id: L1MuonMatcherAlgo.h,v 1.1 2009/05/18 16:38:45 gpetrucc Exp $
//

/**
  \class    L1MuonMatcherAlgo L1MuonMatcherAlgo.h "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMatcherAlgo.h,v 1.1 2009/05/18 16:38:45 gpetrucc Exp $
*/


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
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

    private:
        PropagateToMuon prop_;

        typedef StringCutObjectSelector<l1extra::L1MuonParticle> Selector;
        /// Preselection cut to apply to L1 candidates before matching
        Selector preselectionCut_;

        /// Matching cuts
        double deltaPhi_, deltaR2_;

        /// Sort by deltaPhi instead of deltaR
        bool sortByDeltaPhi_;
};

#endif
