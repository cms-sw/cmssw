// -*- C++ -*-
// $Id: TrajectorySeedFromMuonProducer.cc,v 1.2 2011/12/22 20:27:05 innocent Exp $
//
// Authors: Y.Gao (FNAL)
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// related to reco::Track
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// related to reco::Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// TrajectorSeed Related
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

// TrajectorSeed Related
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "RecoMuon/MuonIdentification/interface/MuonCosmicsId.h"

class TrajectorySeedFromMuonProducer : public edm::EDProducer
{

public:
    explicit TrajectorySeedFromMuonProducer( const edm::ParameterSet & );   
    virtual void produce(edm::Event&, const edm::EventSetup&);

private:
    edm::InputTag  muonCollectionTag_;
    edm::InputTag  trackCollectionTag_;
    bool           skipMatchedMuons_;
};

TrajectorySeedFromMuonProducer::TrajectorySeedFromMuonProducer(const edm::ParameterSet & iConfig)
{
  muonCollectionTag_  = iConfig.getParameter<edm::InputTag>("muonCollectionTag");
  trackCollectionTag_ = iConfig.getParameter<edm::InputTag>("trackCollectionTag");
  skipMatchedMuons_   = iConfig.getParameter<bool>("skipMatchedMuons");
  produces<TrajectorySeedCollection>();
}


void TrajectorySeedFromMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace std;

  // Product
  std::auto_ptr<TrajectorySeedCollection> result(new TrajectorySeedCollection());
  
  edm::ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  
  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
  
  
  
  edm::Handle<edm::View<Muon> > muonCollectionHandle;
  iEvent.getByLabel(muonCollectionTag_, muonCollectionHandle);
  
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(trackCollectionTag_, trackCollectionHandle);

  // Loop over the muon track
  for (edm::View<Muon>::const_iterator muon = muonCollectionHandle->begin();  muon != muonCollectionHandle->end(); ++muon) {
    // muon must have a tracker track
    if( muon->innerTrack().isNull() ) continue;
    edm::RefToBase<reco::Track> track(muon->innerTrack());
    // check if there is a back-to-back track
    if( skipMatchedMuons_ &&
	muonid::findOppositeTrack(trackCollectionHandle, *track).isNonnull() ) continue;
    if( (!track->innerOk()) || (!track->recHit(0)->isValid())) continue;
    GlobalPoint innerPosition(track->innerPosition().x(), track->innerPosition().y(), track->innerPosition().z());
    GlobalVector innerMomentum(track->innerMomentum().x(), track->innerMomentum().y(), track->innerMomentum().z());
    int charge = track->charge();
    const reco::Track::CovarianceMatrix innerStateCovariance = track->innerStateCovariance();
    DetId innerDetId = DetId(track->innerDetId());
    // Construct the information necessary to make a TrajectoryStateOnSurface
    GlobalTrajectoryParameters globalTrajParams(innerPosition, innerMomentum, charge, &(*magneticField));
    CurvilinearTrajectoryError curviError(innerStateCovariance);
    FreeTrajectoryState tracker_state(globalTrajParams, curviError);
    LogTrace("MuonIdentification") << "Track Inner FTS: "<<tracker_state;
    
    TrajectoryStateOnSurface tracker_tsos  = TrajectoryStateOnSurface(globalTrajParams, curviError, trackerGeometry->idToDet(innerDetId)->surface());
    
    // Make Hits, push back the innermost Hit
    edm::OwnVector<TrackingRecHit> trackHits;
    trackHits.push_back(track->recHit(0)->clone() );
    
    // Make TrajectorySeed
    PTrajectoryStateOnDet const & PTraj = trajectoryStateTransform::persistentState(tracker_tsos, innerDetId.rawId());
    TrajectorySeed trajectorySeed(PTraj, trackHits, oppositeToMomentum);
    LogTrace("MuonIdentification")<<"Trajectory Seed Direction: "<< trajectorySeed.direction()<<endl;
    result->push_back(trajectorySeed);
  }
  
  iEvent.put(result);

}

DEFINE_FWK_MODULE(TrajectorySeedFromMuonProducer);
