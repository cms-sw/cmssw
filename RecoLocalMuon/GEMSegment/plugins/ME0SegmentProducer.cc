/** \file ME0SegmentProducer.cc
 *
 */

#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentProducer.h>
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentBuilder.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/GEMRecHit/interface/ME0RecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/ME0Segment.h>

ME0SegmentProducer::ME0SegmentProducer(const edm::ParameterSet& ps) : iev(0) {
	
    theME0RecHitToken = consumes<ME0RecHitCollection>(ps.getParameter<edm::InputTag>("me0RecHitLabel"));
    segmentBuilder_ = std::unique_ptr<ME0SegmentBuilder>(new ME0SegmentBuilder(ps)); // pass on the Parameter Set

    // register what this produces
    produces<ME0SegmentCollection>();
}

ME0SegmentProducer::~ME0SegmentProducer() {}

void ME0SegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

   LogDebug("ME0SegmentProducer") << "start producing segments for " << ++iev << "th event with csc data";
	
    // find the geometry (& conditions?) for this event & cache it in the builder
    edm::ESHandle<ME0Geometry> me0g;
    setup.get<MuonGeometryRecord>().get(me0g);
    const ME0Geometry* mgeom = &*me0g;
    segmentBuilder_->setGeometry(mgeom);
  

    // get the collection of ME0RecHit
    edm::Handle<ME0RecHitCollection> me0RecHits;
    ev.getByToken(theME0RecHitToken,me0RecHits);

    // create empty collection of Segments
    std::auto_ptr<ME0SegmentCollection> oc( new ME0SegmentCollection );

    // fill the collection
    segmentBuilder_->build(me0RecHits.product(), *oc); //@@ FILL oc

    // put collection in event
    ev.put(oc);
}
