// -*- C++ -*-
//
// Package:    TrackerToMuonPropagator
// Class:      TrackerToMuonPropagator
// 
/**\class TrackerToMuonPropagator TrackerToMuonPropagator.cc Alignment/TrackerToMuonPropagator/src/TrackerToMuonPropagator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: TrackerToMuonPropagator.cc,v 1.1 2008/02/14 15:39:58 pivarski Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// references
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

//
// class decleration
//

class TrackerToMuonPropagator : public edm::EDProducer {
   public:
      explicit TrackerToMuonPropagator(const edm::ParameterSet&);
      ~TrackerToMuonPropagator();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      edm::InputTag m_trackerTracks, m_globalMuons;
      std::string m_propagator;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackerToMuonPropagator::TrackerToMuonPropagator(const edm::ParameterSet& iConfig)
{
   m_trackerTracks = iConfig.getParameter<edm::InputTag>("trackerTracks");
   m_globalMuons = iConfig.getParameter<edm::InputTag>("globalMuons");
   m_propagator = iConfig.getParameter<std::string>("propagator");
  
   produces<std::vector<Trajectory> >();
   produces<TrajTrackAssociationCollection>();
}


TrackerToMuonPropagator::~TrackerToMuonPropagator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackerToMuonPropagator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<reco::TrackCollection> trackerTracks;
   iEvent.getByLabel(m_trackerTracks, trackerTracks);

   edm::Handle<reco::TrackCollection> globalMuons;
   iEvent.getByLabel(m_globalMuons, globalMuons);

   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get(m_propagator, propagator);

   edm::ESHandle<TrackerGeometry> trackerGeometry;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

   edm::ESHandle<DTGeometry> dtGeometry;
   iSetup.get<MuonGeometryRecord>().get(dtGeometry);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   edm::ESHandle<MagneticField> magneticField;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

   edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
   iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

   // Create these factories once per event
   TrajectoryStateTransform transformer;
   MuonTransientTrackingRecHitBuilder muonTransBuilder;

   // Create a collection of Trajectories, to put in the Event
   std::auto_ptr<std::vector<Trajectory> > trajectoryCollection(new std::vector<Trajectory>);

   // Remember which trajectory is associated with which track
   std::map<edm::Ref<std::vector<Trajectory> >::key_type, edm::Ref<reco::TrackCollection>::key_type> reference_map;
   edm::Ref<std::vector<Trajectory> >::key_type trajCounter = 0;
   edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;

   for (reco::TrackCollection::const_iterator globalMuon = globalMuons->begin();  globalMuon != globalMuons->end();  ++globalMuon) {
      trackCounter++;

      // find a tracker hit on the globalMuon
      trackingRecHit_iterator globalHit_ontracker = globalMuon->recHitsEnd();
      for (trackingRecHit_iterator hit = globalMuon->recHitsBegin();  hit != globalMuon->recHitsEnd();  ++hit) {
	 if ((*hit)->geographicalId().det() == DetId::Tracker) {
	    globalHit_ontracker = hit;
	    break;
	 }
      }

      // if we found it...
      if (globalHit_ontracker != globalMuon->recHitsEnd()) {
	 unsigned int matchThisId = (*globalHit_ontracker)->geographicalId().rawId();
	 LocalPoint matchThisPos = (*globalHit_ontracker)->localPosition();
	 
	 // search for a tracker track with the same tracker hit
	 reco::TrackCollection::const_iterator trackerTrack_match = trackerTracks->end();
	 for (reco::TrackCollection::const_iterator trackerTrack = trackerTracks->begin();  trackerTrack != trackerTracks->end();  ++trackerTrack) {
	    for (trackingRecHit_iterator hit = trackerTrack->recHitsBegin();  hit != trackerTrack->recHitsEnd();  ++hit) {
	       if ((*hit)->geographicalId().rawId() == matchThisId  &&
		   ((*hit)->localPosition() - matchThisPos).mag() < 1e-10) {
		  trackerTrack_match = trackerTrack;
		  break;
	       } // end if match
	    } // end loop over hits
	 } // end loop over tracker tracks

	 // if we found it...
	 if (trackerTrack_match != trackerTracks->end()) {

	    // get information about the outermost tracker hit
	    GlobalPoint outerPosition(trackerTrack_match->outerPosition().x(), trackerTrack_match->outerPosition().y(), trackerTrack_match->outerPosition().z());
	    GlobalVector outerMomentum(trackerTrack_match->outerMomentum().x(), trackerTrack_match->outerMomentum().y(), trackerTrack_match->outerMomentum().z());
	    int charge = trackerTrack_match->charge();
	    const reco::Track::CovarianceMatrix outerStateCovariance = trackerTrack_match->outerStateCovariance();
	    DetId outerDetId = DetId(trackerTrack_match->outerDetId());

	    // construct the information necessary to make a TrajectoryStateOnSurface
	    GlobalTrajectoryParameters globalTrajParams(outerPosition, outerMomentum, charge, &(*magneticField));
	    CurvilinearTrajectoryError curviError(outerStateCovariance);
	    TrajectoryStateOnSurface state(globalTrajParams, curviError, trackerGeometry->idToDet(outerDetId)->surface());
	    FreeTrajectoryState tracker_state(globalTrajParams, curviError);

	    // starting point for propagation into the muon system
	    TrajectoryStateOnSurface tracker_tsos(tracker_state, trackerGeometry->idToDet(outerDetId)->surface());
	    TrajectoryStateOnSurface last_tsos(tracker_state, trackerGeometry->idToDet(outerDetId)->surface());

	    // loop over the muon hits, keeping track of the successful extrapolations
	    edm::OwnVector<TrackingRecHit> muonHits;
	    std::vector<TrajectoryStateOnSurface> TSOSes;
	    for (trackingRecHit_iterator hit = globalMuon->recHitsBegin();  hit != globalMuon->recHitsEnd();  ++hit) {
	       DetId id = (*hit)->geographicalId();

	       TrajectoryStateOnSurface extrapolation;
	       bool extrapolated = false;
	       if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
		  extrapolation = propagator->propagate(last_tsos, dtGeometry->idToDet(id)->surface());
		  extrapolated = true;
	       }
	       else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
		  extrapolation = propagator->propagate(last_tsos, cscGeometry->idToDet(id)->surface());
		  extrapolated = true;
	       }
	 
	       if (extrapolated  &&  extrapolation.isValid()) {
		  muonHits.push_back((*hit)->clone());
		  TSOSes.push_back(extrapolation);
	       }
	    } // end loop over standAloneMuon hits
	 
	    // if it has any successful extrapolations, make them into a Trajectory
	    if (muonHits.size() > 0) {
	       PTrajectoryStateOnDet *PTraj = transformer.persistentState(tracker_tsos, outerDetId.rawId());
	       TrajectorySeed trajectorySeed(*PTraj, muonHits, alongMomentum);
	       Trajectory trajectory(trajectorySeed, alongMomentum);

	       for (unsigned int i = 0;  i < muonHits.size();  i++) {
		  TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&(muonHits[i]), globalGeometry));
		  TrajectoryStateOnSurface TSOS = TSOSes[i];
		  trajectory.push(TrajectoryMeasurement(TSOS, TSOS, TSOS, hitPtr));
	       } // end filling Trajectory

	       trajectoryCollection->push_back(trajectory);

	       // Remember which Trajectory is associated with which Track
	       trajCounter++;
	       edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;
	       reference_map[trajCounter] = trackCounter;

	    } // end if we have some good extrapolations

	 } // end if we found the tracker track that matches this global track
      } // end if we found any tracker hit on our globalMuon
   } // end loop over globalMuons

   unsigned int numTrajectories = trajectoryCollection->size();

   // insert the trajectories into the Event
   edm::OrphanHandle<std::vector<Trajectory> > ohTrajs = iEvent.put(trajectoryCollection);

   // create the trajectory <-> track association map
   std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap(new TrajTrackAssociationCollection());

   for (trajCounter = 0;  trajCounter < numTrajectories;  trajCounter++) {
      edm::Ref<reco::TrackCollection>::key_type trackCounter = reference_map[trajCounter];

      trajTrackMap->insert(edm::Ref<std::vector<Trajectory> >(ohTrajs, trajCounter), edm::Ref<reco::TrackCollection>(globalMuons, trackCounter));
   }
   // and put it in the Event, also
   iEvent.put(trajTrackMap);
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackerToMuonPropagator::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerToMuonPropagator::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerToMuonPropagator);
