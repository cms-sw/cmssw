#ifndef MuonAnalysis_MuonAssociators_MatcherUsingTracksAlgorithm_h
#define MuonAnalysis_MuonAssociators_MatcherUsingTracksAlgorithm_h
//
// $Id: MatcherUsingTracksAlgorithm.h,v 1.1 2009/06/24 08:47:01 gpetrucc Exp $
//

/**
  \class    pat::MatcherUsingTracksAlgorithm MatcherUsingTracksAlgorithm.h "MuonAnalysis/MuonAssociators/interface/MatcherUsingTracksAlgorithm.h"
  \brief    Matcher of reconstructed objects to other reconstructed objects using the tracks inside them 
            
  \author   Giovanni Petrucciani
  \version  $Id: MatcherUsingTracksAlgorithm.h,v 1.1 2009/06/24 08:47:01 gpetrucc Exp $
*/


#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

class MatcherUsingTracksAlgorithm {
    public:
        explicit MatcherUsingTracksAlgorithm(const edm::ParameterSet & iConfig);
        virtual ~MatcherUsingTracksAlgorithm() { }

        /// Call this method at the beginning of each run, to initialize geometry, magnetic field and propagators
        void init(const edm::EventSetup &iSetup) ;

        /// Try to match one track to another one. Return true if succeeded.
        /// For matches not by ref, it will update deltaR, deltaLocalPos and deltaPtRel if the match suceeded
        bool match(const reco::Candidate &c1, const reco::Candidate &c2, float &deltaR, float &deltaLocalPos, float &deltaPtRel, float &pull) const ;

        /// Find the best match to another candidate, and return its index in the vector
        /// For matches not by ref, it will update deltaR, deltaLocalPos and deltaPtRel if the match suceeded
        /// Returns -1 if the match fails
        int match(const reco::Candidate &tk, const edm::View<reco::Candidate> &c2s, float &deltaR, float &deltaLocalPos, float &deltaPtRel, float &pull) const ;

        /// Return 'true' if the matcher will produce meaningful deltaR, deltaLocalPos, deltaPtRel values
        bool hasMetrics() const { return algo_ != ByTrackRef; }

        /// Return 'true' if the matcher will produce also pull
        bool hasPull() const { return usePull_; }

        /// Compute the pull of two free trajectory states, in the curvilinear frame (q/p, theta, phi, dxy, dsz)
        /// At least one must have errors
        /// diagonalOnly: don't use off-diagonal terms of covariance matrix
        /// useVertex   : use dxy, dsz in the pull (if false, use only q/p, theta, phi)
        /// useFirstMomentum : use the 'start' state momentum to compute dxy, dsx (if false, use 'other')
        static double getPull(const FreeTrajectoryState &start, const FreeTrajectoryState &other, bool diagonalOnly, bool useVertex, bool useFirstMomentum)   ;

        /// Compute the pull of one free trajectory state and a TrajectoryStateClosestToPoint closest to it, in the perigee frame
        /// At least one must have errors
        /// diagonalOnly: don't use off-diagonal terms of covariance matrix
        /// useVertex   : use dxy, dsz in the pull (if false, use only q/p, theta, phi)
        static double getPull(const FreeTrajectoryState &start, const TrajectoryStateClosestToPoint &other, bool diagonalOnly, bool useVertex) ;

        /// Compute the pull of two free trajectory states, in the local frame (q/p, dx, dy, dxdz, dydz)
        /// At least one must have errors
        /// diagonalOnly: don't use off-diagonal terms of covariance matrix
        /// useVertex   : use dx, dy in the pull (if false, use only direction and q/p)
        static double getPull(const TrajectoryStateOnSurface &start, const TrajectoryStateOnSurface &other, bool diagonalOnly, bool usePosition) ;

        /// Possibly crop the 3x3 part of the matrix or remove off-diagonal terms, then invert.
        static void cropAndInvert(AlgebraicSymMatrix55 &cov, bool diagonalOnly, bool top3by3only) ;
    private:
        enum AlgoType   { ByTrackRef, ByDirectComparison, 
                          ByPropagatingSrc, ByPropagatingMatched,
                          ByPropagatingSrcTSCP, ByPropagatingMatchedTSCP }; // propagate closest to point
        enum WhichTrack { None, TrackerTk, MuonTk, GlobalTk };
        enum WhichState { AtVertex, Innermost, Outermost };

        AlgoType      algo_;
        WhichTrack    whichTrack1_, whichTrack2_;
        WhichState    whichState1_, whichState2_;

        // Matching cuts
        float maxLocalPosDiff_;
        float maxGlobalMomDeltaR_;
        float maxGlobalDPtRel_;
        float maxPull_;
        bool  usePull_, pullUseVertex_, pullDiagonalOnly_, pullFirstMomentum_;
        enum SortBy { LocalPosDiff, GlobalMomDeltaR, GlobalDPtRel, Pull};
        SortBy sortBy_;

        //- Tools
        edm::ESHandle<MagneticField>          magfield_;
        edm::ESHandle<Propagator>             propagator_;
        edm::ESHandle<GlobalTrackingGeometry> geometry_;

        /// Get track reference out of a Candidate (via dynamic_cast to reco::RecoCandidate)
        reco::TrackRef getTrack(const reco::Candidate &reco, WhichTrack which) const ;

        /// Starting state for the propagation
        FreeTrajectoryState    startingState(const reco::Candidate &reco, WhichTrack whichTrack, WhichState whichState) const ;

        /// End state for the propagation
        TrajectoryStateOnSurface targetState(const reco::Candidate &reco, WhichTrack whichTrack, WhichState whichState) const ;

        /// Propagate and match. return true if current pair is the new best match (in that case, update also deltaR and deltaLocalPos)
        /// Uses TrajectoryStateClosestToPointBuilder
        bool matchWithPropagation(const FreeTrajectoryState &start, const FreeTrajectoryState &target, 
                float &lastDeltaR, float &lastDeltaLocalPos, float &lastGlobalDPtRel, float &lastPull) const ;

        /// Propagate and match. return true if current pair is the new best match (in that case, update also deltaR and deltaLocalPos)
        /// Uses standard propagator to reach target's surface
        bool matchWithPropagation(const FreeTrajectoryState &start, const TrajectoryStateOnSurface &target, 
                float &lastDeltaR, float &lastDeltaLocalPos, float &lastGlobalDPtRel, float &lastPull) const ;

        /// Compare directly two states. return true if current pair is the new best match (in that case, update also deltaR and deltaLocalPos)
        bool matchByDirectComparison(const FreeTrajectoryState &start, const FreeTrajectoryState &other, 
                float &lastDeltaR, float &lastDeltaLocalPos, float &lastGlobalDPtRel, float &lastPull) const ;

        /// Parse some configuration
        void getConf(const edm::ParameterSet & iConfig, const std::string &whatFor, WhichTrack &whichTrack, WhichState &whichState) ;
   
};


#endif
