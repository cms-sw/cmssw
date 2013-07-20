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
// $Id: TrackerToMuonPropagator.cc,v 1.5 2011/12/22 20:10:49 innocent Exp $
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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
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
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

//
// class decleration
//

class TrackerToMuonPropagator : public edm::EDProducer {
   public:
      explicit TrackerToMuonPropagator(const edm::ParameterSet&);
      ~TrackerToMuonPropagator();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      edm::InputTag m_globalMuons, m_globalMuonTracks;
      std::string m_propagator;

      bool m_refitTracker;
      TrackTransformer *m_trackTransformer;
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
   m_globalMuons = iConfig.getParameter<edm::InputTag>("globalMuons");
   m_globalMuonTracks = iConfig.getParameter<edm::InputTag>("globalMuonTracks");
   m_propagator = iConfig.getParameter<std::string>("propagator");
   m_refitTracker = iConfig.getParameter<bool>("refitTrackerTrack");
   if (m_refitTracker) {
      m_trackTransformer = new TrackTransformer(iConfig.getParameter<edm::ParameterSet>("trackerTrackTransformer"));
   }
   else m_trackTransformer = NULL;
  
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
   if (m_trackTransformer) m_trackTransformer->setServices(iSetup);

   edm::Handle<reco::MuonCollection> globalMuons;
   iEvent.getByLabel(m_globalMuons, globalMuons);

   edm::Handle<reco::TrackCollection> globalMuonTracks;
   iEvent.getByLabel(m_globalMuonTracks, globalMuonTracks);

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
   
   MuonTransientTrackingRecHitBuilder muonTransBuilder;

   // Create a collection of Trajectories, to put in the Event
   std::auto_ptr<std::vector<Trajectory> > trajectoryCollection(new std::vector<Trajectory>);

   // Remember which trajectory is associated with which track
   std::map<edm::Ref<std::vector<Trajectory> >::key_type, edm::Ref<reco::TrackCollection>::key_type> reference_map;
   edm::Ref<std::vector<Trajectory> >::key_type trajCounter = 0;

   for (reco::MuonCollection::const_iterator globalMuon = globalMuons->begin();  globalMuon != globalMuons->end();  ++globalMuon) {

      // get the counter for this global muon (that's why we needed to extract the collection explicitly
      edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;
      reco::TrackCollection::const_iterator globalMuonTrack = globalMuonTracks->begin();
      for (; globalMuonTrack != globalMuonTracks->end();  ++globalMuonTrack) {
	 trackCounter++;
	 if (fabs(globalMuon->combinedMuon()->phi() - globalMuonTrack->phi()) < 1e-10  &&
	     fabs(globalMuon->combinedMuon()->eta() - globalMuonTrack->eta()) < 1e-10) break;
      }
      if (globalMuonTrack == globalMuonTracks->end()) {
	 throw cms::Exception("BadConfig") << "The tracks label doesn't correspond to the same objects as the muons label" << std::endl;
      }

      TrajectoryStateOnSurface tracker_tsos;
      DetId outerDetId;
      if (m_refitTracker) {
	 std::vector<Trajectory> trackerTrajectories = m_trackTransformer->transform(*globalMuon->track());
	 if (trackerTrajectories.size() == 1) {
	    const Trajectory trackerTrajectory = *(trackerTrajectories.begin());

	    // surprisingly, firstMeasurement() corresponds to the outermost state of the tracker
	    tracker_tsos = trackerTrajectory.firstMeasurement().forwardPredictedState();
	    outerDetId = trackerTrajectory.firstMeasurement().recHit()->geographicalId();
	 }
	 else continue;
      }
      else {
	 // get information about the outermost tracker hit
	 GlobalPoint outerPosition(globalMuon->track()->outerPosition().x(), globalMuon->track()->outerPosition().y(), globalMuon->track()->outerPosition().z());
	 GlobalVector outerMomentum(globalMuon->track()->outerMomentum().x(), globalMuon->track()->outerMomentum().y(), globalMuon->track()->outerMomentum().z());
	 int charge = globalMuon->track()->charge();
	 const reco::Track::CovarianceMatrix outerStateCovariance = globalMuon->track()->outerStateCovariance();
	 outerDetId = DetId(globalMuon->track()->outerDetId());

	 // construct the information necessary to make a TrajectoryStateOnSurface
	 GlobalTrajectoryParameters globalTrajParams(outerPosition, outerMomentum, charge, &(*magneticField));
	 CurvilinearTrajectoryError curviError(outerStateCovariance);
	 FreeTrajectoryState tracker_state(globalTrajParams, curviError);

	 // starting point for propagation into the muon system
	 tracker_tsos = TrajectoryStateOnSurface(globalTrajParams, curviError, trackerGeometry->idToDet(outerDetId)->surface());
      }

      TrajectoryStateOnSurface last_tsos = tracker_tsos;

      // loop over the muon hits, keeping track of the successful extrapolations
      edm::OwnVector<TrackingRecHit> muonHits;
      std::vector<TrajectoryStateOnSurface> TSOSes;
      for (trackingRecHit_iterator hit = globalMuon->combinedMuon()->recHitsBegin();  hit != globalMuon->combinedMuon()->recHitsEnd();  ++hit) {
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
	 PTrajectoryStateOnDet const & PTraj = trajectoryStateTransform::persistentState(tracker_tsos, outerDetId.rawId());
	 TrajectorySeed trajectorySeed(PTraj, muonHits, alongMomentum);
	 Trajectory trajectory(trajectorySeed, alongMomentum);

	 for (unsigned int i = 0;  i < muonHits.size();  i++) {
	    TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&(muonHits[i]), globalGeometry));
	    TrajectoryStateOnSurface TSOS = TSOSes[i];
	    trajectory.push(TrajectoryMeasurement(TSOS, TSOS, TSOS, hitPtr));
	 } // end filling Trajectory

	 trajectoryCollection->push_back(trajectory);

	 // Remember which Trajectory is associated with which Track
	 trajCounter++;
	 reference_map[trajCounter] = trackCounter;

      } // end if we have some good extrapolations

   } // end loop over globalMuons

   unsigned int numTrajectories = trajectoryCollection->size();

   // insert the trajectories into the Event
   edm::OrphanHandle<std::vector<Trajectory> > ohTrajs = iEvent.put(trajectoryCollection);

   // create the trajectory <-> track association map
   std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap(new TrajTrackAssociationCollection());

   for (trajCounter = 0;  trajCounter < numTrajectories;  trajCounter++) {
      edm::Ref<reco::TrackCollection>::key_type trackCounter = reference_map[trajCounter];

      trajTrackMap->insert(edm::Ref<std::vector<Trajectory> >(ohTrajs, trajCounter), edm::Ref<reco::TrackCollection>(globalMuonTracks, trackCounter));
   }
   // and put it in the Event, also
   iEvent.put(trajTrackMap);
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackerToMuonPropagator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerToMuonPropagator::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerToMuonPropagator);
