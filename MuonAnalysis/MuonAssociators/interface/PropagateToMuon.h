#ifndef MuonAnalysis_MuonAssociators_interface_PropagateToMuon_h
#define MuonAnalysis_MuonAssociators_interface_PropagateToMuon_h
//
// $Id: PropagateToMuon.h,v 1.4 2011/02/10 00:37:34 gpetrucc Exp $
//

/**
  \class    PropagateToMuon PropagateToMuon.h "MuonAnalysis/MuonAssociators/interface/PropagateToMuon.h"
  \brief    Propagate an object (usually a track) to the second muon station.
            Support for other muon stations will be added on request.
            
  \author   Giovanni Petrucciani
  \version  $Id: PropagateToMuon.h,v 1.4 2011/02/10 00:37:34 gpetrucc Exp $
*/


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
struct DetLayer; // #include "TrackingTools/DetLayers/interface/DetLayer.h" // forward declaration can suffice
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"


class PropagateToMuon {
    public:
        explicit PropagateToMuon(const edm::ParameterSet & iConfig) ;
        ~PropagateToMuon() ;

        /// Call this method at the beginning of each run, to initialize geometry, magnetic field and propagators
        void init(const edm::EventSetup &iSetup) ;

        /// Extrapolate reco::Track to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const reco::Track &tk)     const { return extrapolate(startingState(tk)); }

        /// Extrapolate reco::Candidate to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const reco::Candidate &tk) const { return extrapolate(startingState(tk)); }

        /// Extrapolate a SimTrack to the muon station 2, return an invalid TSOS if it fails; needs the SimVertices too, to know where to start from
        /// Note: it will throw an exception if the SimTrack has no vertex.
        //  don't ask me why SimVertexContainer is in edm namespace
        TrajectoryStateOnSurface extrapolate(const SimTrack &tk, const edm::SimVertexContainer &vtxs) const { return extrapolate(startingState(tk, vtxs)); }

        /// Extrapolate a FreeTrajectoryState to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState &state) const ;
    private:
        enum WhichTrack { None, TrackerTk, MuonTk, GlobalTk };
        enum WhichState { AtVertex, Innermost, Outermost };

        /// Use simplified geometry (cylinders and disks, not individual chambers)
        bool useSimpleGeometry_;

        /// Propagate to MB2 (default) instead of MB1
        bool useMB2_;

        /// Fallback to ME1 if propagation to ME2 fails
        bool fallbackToME1_;

        /// Labels for input collections
        WhichTrack whichTrack_;
        WhichState whichState_;

        /// for cosmics, some things change: the along-opposite is not in-out, nor the innermost/outermost states are in-out really
        bool cosmicPropagation_;

        // needed services for track propagation
        edm::ESHandle<MagneticField> magfield_;
        edm::ESHandle<Propagator> propagator_, propagatorAny_, propagatorOpposite_;
        edm::ESHandle<MuonDetLayerGeometry> muonGeometry_;
        // simplified geometry for track propagation
        const  BoundCylinder *barrelCylinder_;
        const  BoundDisk *endcapDiskPos_[3], *endcapDiskNeg_[3];
        double barrelHalfLength_;
        std::pair<float,float> endcapRadii_[3];

        /// Starting state for the propagation
        FreeTrajectoryState startingState(const reco::Candidate &reco) const ;

        /// Starting state for the propagation
        FreeTrajectoryState startingState(const reco::Track &tk) const ;

        /// Starting state for the propagation
        FreeTrajectoryState startingState(const SimTrack &tk, const edm::SimVertexContainer &vtxs) const ;

        /// Get the best TSOS on one of the chambres of this DetLayer, or an invalid TSOS if none match
        TrajectoryStateOnSurface getBestDet(const TrajectoryStateOnSurface &tsos, const DetLayer *station) const;
};

#endif
