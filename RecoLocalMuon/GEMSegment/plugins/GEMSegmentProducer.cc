/** \class GEMSegmentProducer derived by CSCSegmentProducer 
 * Produces a collection of GEMSegment's in endcap muon GEMs. 
 *
 * \author Piet Verwilligen
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"

#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilder.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class GEMSegmentProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit GEMSegmentProducer(const edm::ParameterSet&);
  /// Destructor
  virtual ~GEMSegmentProducer() {}
  /// Produce the GEMSegment collection
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  int iev; // events through
  edm::EDGetTokenT<GEMRecHitCollection> theGEMRecHitToken;
  std::unique_ptr<GEMSegmentBuilder> segmentBuilder_;
};

GEMSegmentProducer::GEMSegmentProducer(const edm::ParameterSet& ps) : iev(0) {
	
  theGEMRecHitToken = consumes<GEMRecHitCollection>(ps.getParameter<edm::InputTag>("gemRecHitLabel"));
  segmentBuilder_ = std::make_unique<GEMSegmentBuilder>(ps); // pass on the Parameter Set

  // register what this produces
  produces<GEMSegmentCollection>();
}

void GEMSegmentProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {

  LogDebug("GEMSegmentProducer") << "start producing segments for " << ++iev << "th event with GEM data";
	
  // find the geometry (& conditions?) for this event & cache it in the builder
  edm::ESHandle<GEMGeometry> gemg;
  setup.get<MuonGeometryRecord>().get(gemg);
  const GEMGeometry* mgeom = &*gemg;
  segmentBuilder_->setGeometry(mgeom);
  

  // get the collection of GEMRecHit
  edm::Handle<GEMRecHitCollection> gemRecHits;
  ev.getByToken(theGEMRecHitToken,gemRecHits);

  // create empty collection of Segments
  auto oc = std::make_unique<GEMSegmentCollection>();

  // fill the collection
  segmentBuilder_->build(gemRecHits.product(), *oc); //@@ FILL oc

  // put collection in event
  ev.put(std::move(oc));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMSegmentProducer);
