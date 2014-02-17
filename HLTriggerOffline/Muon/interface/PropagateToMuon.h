#ifndef MuonAnalysis_MuonAssociators_interface_PropagateToMuon_h
#define MuonAnalysis_MuonAssociators_interface_PropagateToMuon_h
//
// $Id: PropagateToMuon.h,v 1.1 2010/04/15 18:37:17 klukas Exp $
//

/**
  \class    PropagateToMuon PropagateToMuon.h "HLTriggerOffline/Muon/interface/PropagateToMuon.h"
  \brief    Propagate an object (usually a track) to the second muon station.
            Support for other muon stations will be added on request.
            
  \author   Giovanni Petrucciani
  \version  $Id: PropagateToMuon.h,v 1.1 2010/04/15 18:37:17 klukas Exp $
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

        /// Extrapolate a FreeTrajectoryState to the muon station 2, return an invalid TSOS if it fails
        TrajectoryStateOnSurface extrapolate(const FreeTrajectoryState &state) const ;

    private:
        enum WhichTrack { None, TrackerTk, MuonTk, GlobalTk };
        enum WhichState { AtVertex, Innermost, Outermost };

        /// Labels for input collections
        bool useSimpleGeometry_;
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
