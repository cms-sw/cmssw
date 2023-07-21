/** \class GE0TriggerPseudoProducer
 *
 * Takes offline GE0 segment as input
 * Produces GE0 trigger objects
 *
 * \author Original ME0 code by Tao Huang (TAMU). Converted and updated to GE0 by Ian J. Watson (USeoul)
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TGEM/interface/GE0TriggerPseudoBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GE0TriggerPseudoBuilder;

class GE0TriggerPseudoProducer : public edm::global::EDProducer<> {
public:
  explicit GE0TriggerPseudoProducer(const edm::ParameterSet&);
  ~GE0TriggerPseudoProducer() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::InputTag me0segmentProducer_;
  edm::EDGetTokenT<GEMSegmentCollection> me0segment_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> me0_geom_token_;
  edm::ParameterSet config_;
};

GE0TriggerPseudoProducer::GE0TriggerPseudoProducer(const edm::ParameterSet& conf) {
  me0segmentProducer_ = conf.getParameter<edm::InputTag>("ME0SegmentProducer");
  me0segment_token_ = consumes<GEMSegmentCollection>(me0segmentProducer_);
  me0_geom_token_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  config_ = conf;

  // register what this produces
  produces<GE0TriggerDigiCollection>();
}

GE0TriggerPseudoProducer::~GE0TriggerPseudoProducer() {}

void GE0TriggerPseudoProducer::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& setup) const {
  edm::ESHandle<GEMGeometry> h_me0 = setup.getHandle(me0_geom_token_);

  edm::Handle<GEMSegmentCollection> me0Segmentcoll;
  ev.getByToken(me0segment_token_, me0Segmentcoll);
  const GEMSegmentCollection* me0segments = me0Segmentcoll.product();

  // Create empty collection
  auto oc_trig = std::make_unique<GE0TriggerDigiCollection>();

  auto trigBuilder = std::make_unique<GE0TriggerPseudoBuilder>(config_);
  trigBuilder->setME0Geometry(&*h_me0);

  // Fill output collections if valid input collection is available.
  if (me0Segmentcoll.isValid()) {
    trigBuilder->build(me0segments, *oc_trig);
  }

  // Put collections in event.
  ev.put(std::move(oc_trig));
}

DEFINE_FWK_MODULE(GE0TriggerPseudoProducer);
