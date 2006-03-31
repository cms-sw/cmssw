#include <RecoLocalMuon/CSCSegment/src/CSCSegmentProducer.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilder.h>

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

CSCSegmentProducer::CSCSegmentProducer(const edm::ParameterSet& pas) : iev(0) {
	
    recHitProducer_ = pas.getParameter<std::string>("CSCRecHit2DProducer");
    segmentBuilder_ = new CSCSegmentBuilder(pas); // pass on the PS

  	// register what this produces
    produces<CSCSegmentCollection>();
}

CSCSegmentProducer::~CSCSegmentProducer() {

    LogDebug("CSC") << "deleting segmentBuilder_ after " << iev << " events.\n";
    delete segmentBuilder_;
}

void CSCSegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

    LogDebug("CSC") << "Start producing segments for event " << ++iev << "\n";
	
    // find the geometry (& conditions?) for this event & cache it in the builder
  
    edm::ESHandle<CSCGeometry> h;
    setup.get<MuonGeometryRecord>().get(h);
    const CSCGeometry* pgeom = &*h;
    segmentBuilder_->setGeometry(pgeom);
	
    // get the collection of CSCRecHit2D
    edm::Handle<CSCRecHit2DCollection> cscRecHits;
    ev.getByLabel(recHitProducer_, cscRecHits);  

    // create empty collection of Segments
    std::auto_ptr<CSCSegmentCollection> oc( new CSCSegmentCollection );

  	// fill the collection
    segmentBuilder_->build(cscRecHits.product(), *oc); //@@ FILL oc

    // put collection in event
    ev.put(oc);
}
