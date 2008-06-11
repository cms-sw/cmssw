// -*- C++ -*-
//
// Package:    MuonHIPOverlapsSegmentFilter
// Class:      MuonHIPOverlapsSegmentFilter
// 
/**\class MuonHIPOverlapsSegmentFilter MuonHIPOverlapsSegmentFilter.cc Alignment/MuonHIPOverlapsSegmentFilter/src/MuonHIPOverlapsSegmentFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Mon Jun  9 17:27:41 CEST 2008
// $Id: MuonHIPOverlapsSegmentFilter.cc,v 1.1 2008/06/09 19:48:40 pivarski Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// references
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

//
// class declaration
//

class MuonHIPOverlapsSegmentFilter : public edm::EDFilter {
   public:
      explicit MuonHIPOverlapsSegmentFilter(const edm::ParameterSet&);
      ~MuonHIPOverlapsSegmentFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_input;
      int m_station;
      int m_maxPairs;
      bool m_debuggingHistograms;
      TH1F *th1f_numPairs;

      unsigned long m_total_events, m_segments_on_station, m_segments_on_neighbors, m_onlyN;
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
MuonHIPOverlapsSegmentFilter::MuonHIPOverlapsSegmentFilter(const edm::ParameterSet& iConfig)
   : m_input(iConfig.getParameter<edm::InputTag>("input"))
   , m_station(iConfig.getParameter<int>("station"))
   , m_maxPairs(iConfig.getParameter<int>("maxPairs"))
   , m_debuggingHistograms(iConfig.getUntrackedParameter<bool>("debuggingHistograms", false))
{
   //now do what ever initialization is needed
   produces<reco::TrackCollection>();
   produces<reco::TrackExtraCollection>();
   produces<TrackingRecHitCollection>();

   m_total_events = 0;
   m_segments_on_station = 0;
   m_segments_on_neighbors = 0;
   m_onlyN = 0;

   if (m_debuggingHistograms) {
      edm::Service<TFileService> tfile;
      th1f_numPairs = tfile->make<TH1F>("numPairs", "numPairs", 31, -0.5, 30.5);
   }
}


MuonHIPOverlapsSegmentFilter::~MuonHIPOverlapsSegmentFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MuonHIPOverlapsSegmentFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   m_total_events++;

   edm::Handle<CSCSegmentCollection> segments;
   iEvent.getByLabel(m_input, segments);
   
   std::auto_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection);
   std::auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection);
   std::auto_ptr<TrackingRecHitCollection> trackingRecHitCollection(new TrackingRecHitCollection);

   int segments_on_station = 0;
   for (CSCSegmentCollection::const_iterator segment = segments->begin();  segment != segments->end();  ++segment) {
      if ((segment->cscDetId().endcap() == 1 ? 1 : -1) * segment->cscDetId().station() == m_station) segments_on_station++;
   }

   if (segments_on_station > 0) m_segments_on_station++;

   if (segments_on_station < 2) {
      iEvent.put(trackCollection);
      iEvent.put(trackExtraCollection);
      iEvent.put(trackingRecHitCollection);
      return false;
   }
   // from this point on, nothing needs to be fast

   reco::TrackExtraRefProd refTrackExtra = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
   TrackingRecHitRefProd refTrackingRecHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
   edm::Ref<reco::TrackExtraCollection>::key_type refTrackExtraIndex = 0;
   edm::Ref<TrackingRecHitCollection>::key_type refTrackingRecHitsIndex = 0;

   bool neighbors_found = false;
   int numPairs = 0;
   for (CSCSegmentCollection::const_iterator segment = segments->begin();  segment != segments->end();  ++segment) {
      CSCDetId cscId = segment->cscDetId();
      if ((cscId.endcap() == 1 ? 1 : -1) * cscId.station() == m_station) {

	 int chamber = cscId.chamber();
	 int next = chamber + 1;
	 // Some rings have 36 chambers, others have 18.  This will still be valid when ME4/2 is added.
	 if (next == 37  &&  (abs(cscId.station()) == 1  ||  cscId.ring() == 2)) next = 1;
	 if (next == 19  &&  (abs(cscId.station()) != 1  &&  cscId.ring() == 1)) next = 1;

	 for (CSCSegmentCollection::const_iterator segment2 = segments->begin();  segment2 != segments->end();  ++segment2) {
	    CSCDetId cscId2 = segment2->cscDetId();

	    if ((cscId2.endcap() == 1 ? 1 : -1) * cscId2.station() == m_station  &&  cscId2.chamber() == next) {
	       neighbors_found = true;
	    
	       reco::Track *track = new reco::Track();
	       reco::TrackExtra *trackExtra = new reco::TrackExtra();
	       track->setExtra(reco::TrackExtraRef(refTrackExtra, refTrackExtraIndex++));

	       std::vector<const TrackingRecHit*> hits_on_segment1 = segment->recHits();
	       std::vector<const TrackingRecHit*> hits_on_segment2 = segment2->recHits();
	       std::vector<const TrackingRecHit*> hits_on_segments;

	       for (std::vector<const TrackingRecHit*>::const_iterator hit = hits_on_segment1.begin();  hit != hits_on_segment1.end();  ++hit)
		  hits_on_segments.push_back(*hit);
	       for (std::vector<const TrackingRecHit*>::const_iterator hit = hits_on_segment2.begin();  hit != hits_on_segment2.end();  ++hit)
		  hits_on_segments.push_back(*hit);

	       for (std::vector<const TrackingRecHit*>::const_iterator hit = hits_on_segments.begin();  hit != hits_on_segments.end();  ++hit) {
		  TrackingRecHit *myhit = (*hit)->clone();
		  trackingRecHitCollection->push_back(myhit);
		  trackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));

	       } // end loop over all hits on segments

	       trackCollection->push_back(*track);
	       trackExtraCollection->push_back(*trackExtra);
	       numPairs++;

	    } // end we found a next-door neighbor
	 } // end second loop over segments
      } // end if this segment is on our station
   } // end first loop over segments

   iEvent.put(trackCollection);
   iEvent.put(trackExtraCollection);
   iEvent.put(trackingRecHitCollection);

   if (!neighbors_found) return false;
   m_segments_on_neighbors++;

   if (m_debuggingHistograms) {
      th1f_numPairs->Fill(numPairs);
   }

   if (numPairs > m_maxPairs) return false;

   m_onlyN++;
   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonHIPOverlapsSegmentFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonHIPOverlapsSegmentFilter::endJob() {
   std::cout << "MuonHIPOverlapsSegmentFilter. Total events: " << m_total_events
	     << " events with segments on station " << m_station << ": " << m_segments_on_station
	     << " events with next-door neighbors " << m_segments_on_neighbors << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonHIPOverlapsSegmentFilter);
