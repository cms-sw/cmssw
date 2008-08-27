// -*- C++ -*-
//
// Package:    ZeroFieldStandAloneMuonHIPTrackJoiner
// Class:      ZeroFieldStandAloneMuonHIPTrackJoiner
// 
/**\class ZeroFieldStandAloneMuonHIPTrackJoiner ZeroFieldStandAloneMuonHIPTrackJoiner.cc Alignment/ZeroFieldStandAloneMuonHIPTrackJoiner/src/ZeroFieldStandAloneMuonHIPTrackJoiner.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: ZeroFieldStandAloneMuonHIPTrackJoiner.cc,v 1.1 2008/06/18 22:19:33 pivarski Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// references
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"

// #include "FWCore/ServiceRegistry/interface/Service.h"
// #include "PhysicsTools/UtilAlgos/interface/TFileService.h"
// #include "TH1F.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

class ZeroFieldStandAloneMuonHIPTrackJoiner : public edm::EDProducer {
   public:
      explicit ZeroFieldStandAloneMuonHIPTrackJoiner(const edm::ParameterSet&);
      ~ZeroFieldStandAloneMuonHIPTrackJoiner();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_tracks, m_segments;
      int m_minXmeasuring, m_minZmeasuring;
      unsigned int m_minCSChits;
      double m_minPdot;
      int m_station1, m_station2;
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
ZeroFieldStandAloneMuonHIPTrackJoiner::ZeroFieldStandAloneMuonHIPTrackJoiner(const edm::ParameterSet& iConfig)
   : m_tracks(iConfig.getParameter<edm::InputTag>("tracks"))
   , m_segments(iConfig.getParameter<edm::InputTag>("segments"))
   , m_minXmeasuring(iConfig.getUntrackedParameter<int>("minXmeasuring", 4))
   , m_minZmeasuring(iConfig.getUntrackedParameter<int>("minZmeasuring", 4))
   , m_minCSChits(iConfig.getParameter<unsigned int>("minCSChits"))
   , m_minPdot(iConfig.getParameter<double>("minPdot"))
   , m_station1(iConfig.getParameter<int>("station1"))
   , m_station2(iConfig.getParameter<int>("station2"))
{
   produces<reco::TrackCollection>();
   produces<reco::TrackExtraCollection>();
   produces<TrackingRecHitCollection>();

   if (m_station1 >= m_station2) throw cms::Exception("We must have station1 < station2");
}


ZeroFieldStandAloneMuonHIPTrackJoiner::~ZeroFieldStandAloneMuonHIPTrackJoiner()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ZeroFieldStandAloneMuonHIPTrackJoiner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(m_tracks, tracks);

   edm::Handle<CSCSegmentCollection> segments;
   iEvent.getByLabel(m_segments, segments);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   std::auto_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection);
   std::auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection);
   std::auto_ptr<TrackingRecHitCollection> trackingRecHitCollection(new TrackingRecHitCollection);

   reco::TrackExtraRefProd refTrackExtra = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
   TrackingRecHitRefProd refTrackingRecHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
   edm::Ref<reco::TrackExtraCollection>::key_type refTrackExtraIndex = 0;
   edm::Ref<TrackingRecHitCollection>::key_type refTrackingRecHitsIndex = 0;

   if (m_station1 == 0) {
      for (reco::TrackCollection::const_iterator track1 = tracks->begin();  track1 != tracks->end();  ++track1) {
	 int xmeasuring = 0;
	 int zmeasuring = 0;
	 for (trackingRecHit_iterator hit = track1->recHitsBegin();  hit != track1->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();
	    if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	       DTLayerId dtId(id.rawId());
	       
	       if (dtId.superlayer() == 2) zmeasuring++;
	       else xmeasuring++;
	    }
	 }

	 if (xmeasuring >= m_minXmeasuring  &&  zmeasuring >= m_minZmeasuring) {
	    math::XYZVector momentum1 = track1->momentum();

	    std::cout << "momentum1 " << momentum1 << std::endl;

	    math::XYZVector momentum2;
	    CSCSegmentCollection::const_iterator bestsegment = segments->end();
	    for (CSCSegmentCollection::const_iterator segment2 = segments->begin();  segment2 != segments->end();  ++segment2) {
	       CSCDetId id2(segment2->geographicalId().rawId());
	       int station2 = (id2.endcap() == 1 ? 1 : -1) * id2.station();
	       if (m_station2 == station2  &&  segment2->recHits().size() >= m_minCSChits) {
		  GlobalVector momentum_GlobalVector = cscGeometry->idToDet(id2)->toGlobal(segment2->localDirection());
		  math::XYZVector momentum(momentum_GlobalVector.x(), momentum_GlobalVector.y(), momentum_GlobalVector.z());
		  double momentum_dotproduct = fabs(momentum1.Dot(momentum)) / sqrt(momentum1.Mag2()) / sqrt(momentum.Mag2());

		  std::cout << "    momentum " << momentum << " dotproduct = " << momentum_dotproduct << std::endl;

		  if (bestsegment == segments->end()  ||  momentum_dotproduct > fabs(momentum1.Dot(momentum2)) / sqrt(momentum1.Mag2()) / sqrt(momentum2.Mag2())) {
		     momentum2 = momentum;
		     bestsegment = segment2;
		  }
	       } // end if this is the CSC station we're looking for
	    } // end loop over segments

	    if (bestsegment != segments->end()) {
	       std::cout << "momentum2 = " << momentum2 << " dotproduct = " << fabs(momentum1.Dot(momentum2)) / sqrt(momentum1.Mag2()) / sqrt(momentum2.Mag2()) << std::endl;
	    }
	    
	    if (bestsegment != segments->end()  &&  fabs(momentum1.Dot(momentum2)) / sqrt(momentum1.Mag2()) / sqrt(momentum2.Mag2()) > m_minPdot) {
	       double e[] = { 1.1, 1.2, 2.2, 1.3, 2.3, 3.3, 1.4, 2.4, 3.4, 4.4, 1.5, 2.5, 3.5, 4.5, 5.5 };
	       reco::TrackBase::CovarianceMatrix covarianceMatrix(e, e+15);
	       reco::Track *newtrack = new reco::Track(10., 10, reco::TrackBase::Point(0.,0.,0.), reco::TrackBase::Vector(0.,0.,0.), 1, covarianceMatrix);
	       reco::TrackExtra *newtrackExtra = new reco::TrackExtra(reco::TrackBase::Point(0.,0.,0.), reco::TrackBase::Vector(0.,0.,0.), true, reco::TrackBase::Point(0.,0.,0.), reco::TrackBase::Vector(0.,0.,0.), true, covarianceMatrix, CSCDetId(1, 1, 1, 1, 1), covarianceMatrix, CSCDetId(1, 1, 1, 1, 1), anyDirection, edm::RefToBase<TrajectorySeed>());
	       newtrack->setExtra(reco::TrackExtraRef(refTrackExtra, refTrackExtraIndex++));

// 	       reco::Track *newtrack = new reco::Track(track->chi2(), track->ndof(), track->referencePoint(), track->momentum(), track->charge(), track->covariance(), track->algo(), reco::TrackBase::loose);
// 	       reco::TrackExtra *newtrackExtra = new reco::TrackExtra(track->extra()->outerPosition(), track->extra()->outerMomentum(), track->extra()->outerOk(), track->extra()->innerPosition(), track->extra()->innerMomentum(), track->extra()->innerOk(), track->extra()->outerStateCovariance(), track->extra()->outerDetId(), track->extra()->innerStateCovariance(), track->extra()->innerDetId(), track->extra()->seedDirection(), track->extra()->seedRef());
// 	       newtrack->setExtra(reco::TrackExtraRef(refTrackExtra, refTrackExtraIndex++));

	       for (trackingRecHit_iterator hit = track1->recHitsBegin();  hit != track1->recHitsEnd();  ++hit) {
		  TrackingRecHit *myhit = (*hit)->clone();
		  trackingRecHitCollection->push_back(myhit);
		  newtrackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));
	       }

	       std::vector<const TrackingRecHit*> segmenthits2 = bestsegment->recHits();
	       for (std::vector<const TrackingRecHit*>::const_iterator hit = segmenthits2.begin();  hit != segmenthits2.end();  ++hit) {
		  TrackingRecHit *myhit = (*hit)->clone();
		  trackingRecHitCollection->push_back(myhit);
		  newtrackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));
	       }

	       trackCollection->push_back(*newtrack);
	       trackExtraCollection->push_back(*newtrackExtra);

	    } // end DT track is parallel to CSC segment
	 } // end if enough DT hits
      } // end loop over tracks
   } // end if barrel to endcap

   // endcap to endcap
   else {
      for (CSCSegmentCollection::const_iterator segment1 = segments->begin();  segment1 != segments->end();  ++segment1) {
	 CSCDetId id1(segment1->geographicalId().rawId());
	 int station1 = (id1.endcap() == 1 ? 1 : -1) * id1.station();
	 if (m_station1 == station1  &&  segment1->recHits().size() > m_minCSChits) {
	    GlobalVector momentum1_GlobalVector = cscGeometry->idToDet(id1)->toGlobal(segment1->localDirection());
	    math::XYZVector momentum1(momentum1_GlobalVector.x(), momentum1_GlobalVector.y(), momentum1_GlobalVector.z());

	    math::XYZVector momentum2;
	    CSCSegmentCollection::const_iterator bestsegment = segments->end();
	    for (CSCSegmentCollection::const_iterator segment2 = segments->begin();  segment2 != segments->end();  ++segment2) {
	       CSCDetId id2(segment2->geographicalId().rawId());
	       int station2 = (id2.endcap() == 1 ? 1 : -1) * id2.station();
	       if (m_station2 == station2  &&  segment2->recHits().size() > m_minCSChits) {
		  GlobalVector momentum_GlobalVector = cscGeometry->idToDet(id2)->toGlobal(segment2->localDirection());
		  math::XYZVector momentum(momentum_GlobalVector.x(), momentum_GlobalVector.y(), momentum_GlobalVector.z());
		  double momentum_dotproduct = fabs(momentum1.Dot(momentum)) / sqrt(momentum1.Mag2()) / sqrt(momentum.Mag2());

		  if (bestsegment == segments->end()  ||  momentum_dotproduct > fabs(momentum1.Dot(momentum2)) / sqrt(momentum1.Mag2()) / sqrt(momentum2.Mag2())) {
		     momentum2 = momentum;
		     bestsegment = segment2;
		  }
	       } // end if segment2 is on the right station
	    } // end second loop over segment

	    if (bestsegment != segments->end()  &&  fabs(momentum1.Dot(momentum2)) / sqrt(momentum1.Mag2()) / sqrt(momentum2.Mag2()) > m_minPdot) {
	       double e[] = { 1.1, 1.2, 2.2, 1.3, 2.3, 3.3, 1.4, 2.4, 3.4, 4.4, 1.5, 2.5, 3.5, 4.5, 5.5 };
	       reco::TrackBase::CovarianceMatrix covarianceMatrix(e, e+15);
	       reco::Track *newtrack = new reco::Track(10., 10, reco::TrackBase::Point(0.,0.,0.), reco::TrackBase::Vector(0.,0.,0.), 1, covarianceMatrix);
	       reco::TrackExtra *newtrackExtra = new reco::TrackExtra(reco::TrackBase::Point(0.,0.,0.), reco::TrackBase::Vector(0.,0.,0.), true, reco::TrackBase::Point(0.,0.,0.), reco::TrackBase::Vector(0.,0.,0.), true, covarianceMatrix, CSCDetId(1, 1, 1, 1, 1), covarianceMatrix, CSCDetId(1, 1, 1, 1, 1), anyDirection, edm::RefToBase<TrajectorySeed>());
	       newtrack->setExtra(reco::TrackExtraRef(refTrackExtra, refTrackExtraIndex++));

	       std::vector<const TrackingRecHit*> segmenthits1 = segment1->recHits();
	       for (std::vector<const TrackingRecHit*>::const_iterator hit = segmenthits1.begin();  hit != segmenthits1.end();  ++hit) {
		  TrackingRecHit *myhit = (*hit)->clone();
		  trackingRecHitCollection->push_back(myhit);
		  newtrackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));
	       }

	       std::vector<const TrackingRecHit*> segmenthits2 = bestsegment->recHits();
	       for (std::vector<const TrackingRecHit*>::const_iterator hit = segmenthits2.begin();  hit != segmenthits2.end();  ++hit) {
		  TrackingRecHit *myhit = (*hit)->clone();
		  trackingRecHitCollection->push_back(myhit);
		  newtrackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));
	       }

	       trackCollection->push_back(*newtrack);
	       trackExtraCollection->push_back(*newtrackExtra);

	    } // end if the two CSC segments are parallel

	 } // end if segment1 is on the right station
      } // end first loop over segment
   } // end if endcap to endcap

   iEvent.put(trackCollection);
   iEvent.put(trackExtraCollection);
   iEvent.put(trackingRecHitCollection);
}

// ------------ method called once each job just before starting event loop  ------------
void 
ZeroFieldStandAloneMuonHIPTrackJoiner::beginJob(const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZeroFieldStandAloneMuonHIPTrackJoiner::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeroFieldStandAloneMuonHIPTrackJoiner);
