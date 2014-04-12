#ifndef RecoMuon_MuonIdentification_MuonKinkFinder_h
#define RecoMuon_MuonIdentification_MuonKinkFinder_h

#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

class MuonKinkFinder {
    public:
        MuonKinkFinder(const edm::ParameterSet &iConfig);
        ~MuonKinkFinder() ;

        // set event setup
        void init(const edm::EventSetup &iSetup) ;

        // fill data, return false if refit failed or too few hits
        bool fillTrkKink(reco::MuonQuality & quality, const Trajectory &trajectory) const ;

        // fill data, return false if refit failed or too few hits
        bool fillTrkKink(reco::MuonQuality & quality, const reco::Track &track) const ;

    private:
        /// use only on-diagonal terms of the covariance matrices
        bool diagonalOnly_;
        /// if true, use full 5x5 track state; if false, use only the track direction
        bool usePosition_;

        /// Track Transformer
        TrackTransformer refitter_;

        // compute chi2 between track states
        double getChi2(const TrajectoryStateOnSurface &start, const TrajectoryStateOnSurface &other) const ;

        // possibly crop matrix or set to zero off-diagonal elements, then invert
        void cropAndInvert(AlgebraicSymMatrix55 &cov) const ;
};
#endif
