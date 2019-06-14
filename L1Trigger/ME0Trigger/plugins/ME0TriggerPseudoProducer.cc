#include "L1Trigger/ME0Trigger/plugins/ME0TriggerPseudoProducer.h"
#include "L1Trigger/ME0Trigger/interface/ME0TriggerPseudoBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

ME0TriggerPseudoProducer::ME0TriggerPseudoProducer(const edm::ParameterSet& conf) {
  me0segmentProducer_ = conf.getParameter<edm::InputTag>("ME0SegmentProducer");
  me0segment_token_ = consumes<ME0SegmentCollection>(me0segmentProducer_);
  config_ = conf;

  // register what this produces
  produces<ME0TriggerDigiCollection>();
}

ME0TriggerPseudoProducer::~ME0TriggerPseudoProducer() {}

void ME0TriggerPseudoProducer::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& setup) const {
  edm::ESHandle<ME0Geometry> h_me0;
  setup.get<MuonGeometryRecord>().get(h_me0);

  edm::Handle<ME0SegmentCollection> me0Segmentcoll;
  ev.getByToken(me0segment_token_, me0Segmentcoll);
  const ME0SegmentCollection* me0segments = me0Segmentcoll.product();

  // Create empty collection
  auto oc_trig = std::make_unique<ME0TriggerDigiCollection>();

  auto trigBuilder = std::make_unique<ME0TriggerPseudoBuilder>(config_);
  trigBuilder->setME0Geometry(&*h_me0);

  // Fill output collections if valid input collection is available.
  if (me0Segmentcoll.isValid()) {
    trigBuilder->build(me0segments, *oc_trig);
  }

  // Put collections in event.
  ev.put(std::move(oc_trig));
}
