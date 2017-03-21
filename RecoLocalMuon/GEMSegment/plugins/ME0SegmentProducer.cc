/** \class ME0SegmentProducer derived by CSCSegmentProducer 
 * Produces a collection of ME0Segment's in endcap muon ME0s. 
 *
 * \author Marcello Maggi
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0Segment.h"
#include "DataFormats/Common/interface/Handle.h"

#include "RecoLocalMuon/GEMSegment/plugins/ME0SegmentBuilder.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"


class ME0SegmentProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit ME0SegmentProducer(const edm::ParameterSet&);
  /// Destructor
  virtual ~ME0SegmentProducer() {}
  /// Produce the ME0Segment collection
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  int iev; // events through
  edm::EDGetTokenT<ME0RecHitCollection> theME0RecHitToken;
  std::unique_ptr<ME0SegmentBuilder> segmentBuilder_;
};

ME0SegmentProducer::ME0SegmentProducer(const edm::ParameterSet& ps) : iev(0) {
	
  theME0RecHitToken = consumes<ME0RecHitCollection>(ps.getParameter<edm::InputTag>("me0RecHitLabel"));
  segmentBuilder_ = std::make_unique<ME0SegmentBuilder>(ps); // pass on the Parameter Set

  // register what this produces
  produces<ME0SegmentCollection>();
}

void ME0SegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

  LogDebug("ME0SegmentProducer") << "start producing segments for " << ++iev << "th event with ME0 data";
	
  // find the geometry (& conditions?) for this event & cache it in the builder
  edm::ESHandle<ME0Geometry> me0g;
  setup.get<MuonGeometryRecord>().get(me0g);
  const ME0Geometry* mgeom = &*me0g;
  segmentBuilder_->setGeometry(mgeom);
  

  // get the collection of ME0RecHit
  edm::Handle<ME0RecHitCollection> me0RecHits;
  ev.getByToken(theME0RecHitToken,me0RecHits);

  // create empty collection of Segments
  auto oc = std::make_unique<ME0SegmentCollection>();

  // fill the collection
  segmentBuilder_->build(me0RecHits.product(), *oc); //@@ FILL oc

  // put collection in event
  ev.put(std::move(oc));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ME0SegmentProducer);
