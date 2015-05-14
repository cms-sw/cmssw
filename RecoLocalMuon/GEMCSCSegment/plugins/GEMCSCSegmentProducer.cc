/** \file GEMCSCSegmentProducer.cc
 *
 */

#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentProducer.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentBuilder.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h>
#include <DataFormats/GEMRecHit/interface/GEMCSCSegment.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

GEMCSCSegmentProducer::GEMCSCSegmentProducer(const edm::ParameterSet& pas) : iev(0) {
	
  csc_token = consumes<CSCSegmentCollection>( pas.getParameter<edm::InputTag>("inputObjectsCSC"));
  gem_token = consumes<GEMRecHitCollection> ( pas.getParameter<edm::InputTag>("inputObjectsGEM"));
  segmentBuilder_  = new GEMCSCSegmentBuilder(pas); // pass on the parameterset
  
  // register what this produces
  produces<GEMCSCSegmentCollection>();
}


GEMCSCSegmentProducer::~GEMCSCSegmentProducer() {

    LogDebug("GEMCSCSegment") << "deleting GEMCSCSegmentBuilder after " << iev << " events w/ gem and csc data.";
    delete segmentBuilder_;
}


void GEMCSCSegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {
    LogDebug("GEMCSCSegment") << "start producing segments for " << ++iev << "th event w/ gem and csc data";
	
    // find the geometry (& conditions?) for this event & cache it in the builder
    edm::ESHandle<CSCGeometry> cscg;
    setup.get<MuonGeometryRecord>().get(cscg);
    const CSCGeometry* cgeom = &*cscg;
    
    edm::ESHandle<GEMGeometry> gemg;
    setup.get<MuonGeometryRecord>().get(gemg);
    const GEMGeometry* ggeom = &*gemg;
    
    // cache the geometry in the builder
    segmentBuilder_->setGeometry(ggeom,cgeom);

    // fill the map with matches between GEM and CSC chambers
    segmentBuilder_->LinkGEMRollsToCSCChamberIndex(ggeom,cgeom);

    // get the collection of CSCSegment and GEMRecHits
    edm::Handle<CSCSegmentCollection> cscSegment;
    ev.getByToken(csc_token, cscSegment);
    edm::Handle<GEMRecHitCollection> gemRecHits;
    ev.getByToken(gem_token, gemRecHits);    

    // create empty collection of GEMCSC Segments
    std::auto_ptr<GEMCSCSegmentCollection> oc( new GEMCSCSegmentCollection );

    // pass the empty collection of GEMCSC Segments and fill it
    segmentBuilder_->build(gemRecHits.product(), cscSegment.product(), *oc); //@@ FILL oc
    
    // put the filled collection in event
    ev.put(oc);
}
