#ifndef MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
#define MuonAnalysis_MuonAssociators_interface_L1MuonMatcherAlgo_h
//
// $Id: L1MuonMatcherAlgo.h,v 1.4 2009/05/12 17:20:45 gpetrucc Exp $
//

/**
  \class    L1MuonMatcherAlgo L1MuonMatcherAlgo.h "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"
  \brief    Matcher of reconstructed objects to L1 Muons 
            
  \author   Giovanni Petrucciani
  \version  $Id: L1MuonMatcher.h,v 1.4 2009/05/12 17:20:45 gpetrucc Exp $
*/


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class L1MuonMatcherAlgo {
    public:
        explicit L1MuonMatcherAlgo(const edm::ParameterSet & iConfig) ;
        ~L1MuonMatcherAlgo() ;

        /// Call this method at the beginning of each run, to initialize geometry, magnetic field and propagators
        void init(const edm::EventSetup &iSetup) ;

        /// Extrapolate reco::Track to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const reco::Track &tk)     const { return extrapolate(startingState(tk)); }

        /// Extrapolate reco::Candidate to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const reco::Candidate &tk) const { return extrapolate(startingState(tk)); }

        /// Extrapolate a FreeTrajectoryState to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState &state) const ;

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
        typedef StringCutObjectSelector<l1extra::L1MuonParticle> Selector;
        enum WhichTrack { None, TrackerTk, MuonTk, GlobalTk };
        enum WhichState { AtVertex, Innermost, Outermost };

        /// Labels for input collections
        bool useSimpleGeometry_;
        WhichTrack whichTrack_;
        WhichState whichState_;

        /// Preselection cut to apply to L1 candidates before matching
        Selector preselectionCut_;

        /// Matching cuts
        double deltaPhi_, deltaR2_;

        /// Sort by deltaPhi instead of deltaR
        bool sortByDeltaPhi_;

        /// for cosmics, some things change: the along-opposite is not in-out, nor the innermost/outermost states are in-out really
        bool cosmicPropagation_;

        // needed services for track propagation
        edm::ESHandle<MagneticField> magfield_;
        edm::ESHandle<Propagator> propagator_, propagatorAny_, propagatorOpposite_;
        edm::ESHandle<MuonDetLayerGeometry> muonGeometry_;
        // simplified geometry for track propagation
        const  BoundCylinder *barrelCylinder_;
        const  BoundDisk *endcapDiskPos_, *endcapDiskNeg_;
        double barrelHalfLength_;
        std::pair<float,float> endcapRadii_;

        /// Starting state for the propagation
        FreeTrajectoryState startingState(const reco::Candidate &reco) const ;

        /// Starting state for the propagation
        FreeTrajectoryState startingState(const reco::Track &tk) const ;

        /// Get the best TSOS on one of the chambres of this DetLayer, or an invalid TSOS if none match
        TrajectoryStateOnSurface getBestDet(const TrajectoryStateOnSurface &tsos, const DetLayer *station) const;

};

#endif
