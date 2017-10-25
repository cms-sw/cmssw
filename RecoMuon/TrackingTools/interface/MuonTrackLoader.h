#ifndef RecoMuon_TrackingTools_MuonTrackLoader_H
#define RecoMuon_TrackingTools_MuonTrackLoader_H

/** \class MuonTrackLoader
 *  Class to load the tracks in the event, it provide some common functionalities
 *  both for all the RecoMuon producers.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"


namespace edm {class Event; class EventSetup; class ParameterSet;}

class Trajectory;
class Propagator;
class MuonServiceProxy;
class MuonUpdatorAtVertex;
class TrajectorySmoother;
class ForwardDetLayer;
class BarrelDetLayer;
class TrackerTopology;

class MuonTrackLoader {
  public:

    typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
    typedef MuonCandidate::CandidateContainer CandidateContainer;

    /// Constructor for the STA reco the args must be specify!
    MuonTrackLoader(edm::ParameterSet &parameterSet,edm::ConsumesCollector& iC,  const MuonServiceProxy *service =nullptr);

    /// Destructor
    virtual ~MuonTrackLoader();
   
    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer&, 
                                                        edm::Event&,
                                                        const TrackerTopology& ttopo,
                                                        const std::string& = "", 
							bool = true);

    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer&, 
                                                        edm::Event&, std::vector<bool>&,
                                                        const TrackerTopology& ttopo,
							const std::string& = "",
							bool = true);

    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::TrackCollection> loadTracks(const TrajectoryContainer&, 
                                                        edm::Event&,const std::vector<std::pair<Trajectory*, reco::TrackRef> >&,
                                                        edm::Handle<reco::TrackCollection> const& trackHandle,
                                                        const TrackerTopology& ttopo,
							const std::string& = "", 
							bool = true);

    /// Convert the trajectories into tracks and load the tracks in the event
    edm::OrphanHandle<reco::MuonTrackLinksCollection> loadTracks(const CandidateContainer&,
                                                                 edm::Event&,
                                                                 const TrackerTopology& ttopo);
  
  private:
    static std::vector<const TrackingRecHit*> unpackHit(const TrackingRecHit &hit);

    /// Build a track at the PCA WITHOUT any vertex constriant
    std::pair<bool,reco::Track> buildTrackAtPCA(const Trajectory& trajectory, const reco::BeamSpot &) const;

    /// Takes a track at the PCA and applies the vertex constriant
    std::pair<bool,reco::Track> buildTrackUpdatedAtPCA(const reco::Track& trackAtPCA, const reco::BeamSpot &) const;

    reco::TrackExtra buildTrackExtra(const Trajectory&) const;

    const MuonServiceProxy *theService;

    bool theUpdatingAtVtx;
    MuonUpdatorAtVertex *theUpdatorAtVtx;

    bool theTrajectoryFlag;

    bool theSmoothingStep;
    std::string theSmootherName;
    std::string theTrackerRecHitBuilderName;
    std::unique_ptr<TrajectorySmoother> theSmoother;
    TkClonerImpl hitCloner;


    edm::InputTag theBeamSpotInputTag; 
    edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;

    /// Label for L2SeededTracks
    std::string theL2SeededTkLabel; 
    bool thePutTkTrackFlag;
    bool theSmoothTkTrackFlag;
    bool theAllowNoVtxFlag;
};
#endif
