#ifndef MuonAnalysis_MuonAssociators_src_MatcherByPullsAlgorithm_h
#define MuonAnalysis_MuonAssociators_src_MatcherByPullsAlgorithm_h
// -*- C++ -*-
//
// Package:    MuonAnalysis/MuonAssociators
// Class:      MatcherByPullsAlgorithm
// 
/**\class MatcherByPullsAlgorithm MatcherByPullsAlgorithm.cc MuonAnalysis/MuonAssociators/interface/MatcherByPullsAlgorithm.cc

 Description: Matches a RecoCandidate to a GenParticle (or any other Candidate) using the pulls of the helix parameters

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giovanni Petrucciani (SNS Pisa and CERN PH-CMG)
//         Created:  Sun Nov 16 16:14:09 CET 2008
// $Id: MatcherByPullsAlgorithm.h,v 1.2 2009/11/05 13:20:19 gpetrucc Exp $
//
//

#include <string>
#include <algorithm>
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

//
// class decleration

class MatcherByPullsAlgorithm {
    public:
        explicit MatcherByPullsAlgorithm(const edm::ParameterSet&);
        ~MatcherByPullsAlgorithm();

        /// Match Track to MC Candidate, using already inverted covariance matrix
        /// Return status of match and pull, or (-1,9e9)
        std::pair<bool,float> match(const reco::Track &tk, 
                const reco::Candidate &mc, 
                const AlgebraicSymMatrix55 &invertedCovariance) const ;

        /// Match Reco Candidate to MC Candidates, skipping the ones which are not good
        /// Return index of matchin and pull, or (-1,9e9)
        std::pair<int,float>  match(const reco::RecoCandidate &src, 
                const std::vector<reco::GenParticle> &cands,
                const std::vector<uint8_t>           &good) const ;

        /// Match Reco Candidate to MC Candidates, allowing multiple matches and skipping the ones which are not good
        /// It will fill in the vector of <double,int> with pull and index for all matching candidates,
        /// already sorted by pulls.
        /// This method assumes that matchesToFill is empty when the method is called
        void matchMany(const reco::RecoCandidate &src,
                const std::vector<reco::GenParticle> &cands,
                const std::vector<uint8_t>           &good,
                std::vector<std::pair<double, int> > &matchesToFill) const ;

        /// Match Reco Track to MC Tracks, skipping the ones which are not good
        /// Return index of matchin and pull, or (-1,9e9)
        std::pair<int,float>  match(const reco::Track &src, 
                const std::vector<reco::GenParticle> &cands,
                const std::vector<uint8_t>           &good) const ;

        /// Match Reco Track to MC Tracks, allowing multiple matches and skipping the ones which are not good
        /// It will fill in the vector of <double,int> with pull and index for all matching candidates,
        /// already sorted by pulls.
        /// This method assumes that matchesToFill is empty when the method is called
        void matchMany(const reco::Track &src,
                const std::vector<reco::GenParticle> &cands,
                const std::vector<uint8_t>           &good,
                std::vector<std::pair<double, int> > &matchesToFill) const ;


        /// Fill the inverse covariance matrix for the match(track, candidate, invCov) method
        void fillInvCov(const reco::Track &tk, AlgebraicSymMatrix55 &invCov) const ;
    private:
        /// Get track out of Candidate, NULL if missing
        const reco::Track *   track(const reco::RecoCandidate &src) const ;

        /// Enum to define which track to use
        enum TrackChoice { StaTrack, TrkTrack, GlbTrack };

        /// Track to be used in matching
        TrackChoice track_;

        /// DeltaR of the matching cone
        double dr2_;

        /// Cut on the pull
        double cut_;

        /// Use only the diagonal terms of the covariance matrix
        bool diagOnly_;

        /// Use also dxy / dsz in the matching
        bool useVertex_;  

};

#endif

