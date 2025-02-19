/** \file CSCSegmentProducer.cc
 *
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentProducer.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilder.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

CSCSegmentProducer::CSCSegmentProducer(const edm::ParameterSet& pas) : iev(0) {
	
    inputObjectsTag = pas.getParameter<edm::InputTag>("inputObjects");
    segmentBuilder_ = new CSCSegmentBuilder(pas); // pass on the PS

  	// register what this produces
    produces<CSCSegmentCollection>();
}

CSCSegmentProducer::~CSCSegmentProducer() {

    LogDebug("CSCSegment|CSC") << "deleting CSCSegmentBuilder after " << iev << " events w/csc data.";
    delete segmentBuilder_;
}

void CSCSegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

    LogDebug("CSCSegment|CSC") << "start producing segments for " << ++iev << "th event with csc data";
	
    // find the geometry (& conditions?) for this event & cache it in the builder
  
    edm::ESHandle<CSCGeometry> h;
    setup.get<MuonGeometryRecord>().get(h);
    const CSCGeometry* pgeom = &*h;
    segmentBuilder_->setGeometry(pgeom);
	
    // get the collection of CSCRecHit2D
    edm::Handle<CSCRecHit2DCollection> cscRecHits;
    ev.getByLabel(inputObjectsTag, cscRecHits);  

    // create empty collection of Segments
    std::auto_ptr<CSCSegmentCollection> oc( new CSCSegmentCollection );

  	// fill the collection
    segmentBuilder_->build(cscRecHits.product(), *oc); //@@ FILL oc

    // put collection in event
    ev.put(oc);
}
