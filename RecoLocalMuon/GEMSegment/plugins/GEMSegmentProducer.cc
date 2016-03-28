/** \file GEMSegmentProducer.cc
 *
 */

#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentProducer.h>
#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilder.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMSegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMSegment.h>

GEMSegmentProducer::GEMSegmentProducer(const edm::ParameterSet& ps) : iev(0) {
	
    theGEMRecHitToken = consumes<GEMRecHitCollection>(ps.getParameter<edm::InputTag>("gemRecHitLabel"));
    segmentBuilder_ = std::unique_ptr<GEMSegmentBuilder>(new GEMSegmentBuilder(ps)); // pass on the Parameter Set

    // register what this produces
    produces<GEMSegmentCollection>();
}

GEMSegmentProducer::~GEMSegmentProducer() {}

void GEMSegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

   LogDebug("GEMSegmentProducer") << "start producing segments for " << ++iev << "th event with csc data";
	
    // find the geometry (& conditions?) for this event & cache it in the builder
    edm::ESHandle<GEMGeometry> gemg;
    setup.get<MuonGeometryRecord>().get(gemg);
    const GEMGeometry* mgeom = &*gemg;
    segmentBuilder_->setGeometry(mgeom);
  

    // get the collection of GEMRecHit
    edm::Handle<GEMRecHitCollection> gemRecHits;
    ev.getByToken(theGEMRecHitToken,gemRecHits);

    // create empty collection of Segments
    std::auto_ptr<GEMSegmentCollection> oc( new GEMSegmentCollection );

    // fill the collection
    segmentBuilder_->build(gemRecHits.product(), *oc); //@@ FILL oc

    // put collection in event
    ev.put(oc);
}
