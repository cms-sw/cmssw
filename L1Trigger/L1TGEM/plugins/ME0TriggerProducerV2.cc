/** \class ME0TriggerProducerV2 derived by ME0TriggerProducer
 * Produces a collection of ME0TriggerDigi's in ME0. 
 *
 * \author Woohyeon Heo
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
#include "L1Trigger/L1TGEM/interface/ME0StubBuilderV2.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

class ME0StubBuilderV2;

class ME0TriggerProducerV2 : public edm::global::EDProducer<> {
public:
  explicit ME0TriggerProducerV2(const edm::ParameterSet&);
  ~ME0TriggerProducerV2() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::InputTag me0PadDigis_;
  edm::EDGetTokenT<GEMPadDigiCollection> me0_pad_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> me0_geom_token_;
  edm::ParameterSet config_;
};

ME0TriggerProducerV2::ME0TriggerProducerV2(const edm::ParameterSet& conf) {
  me0PadDigis_ = conf.getParameter<edm::InputTag>("ME0PadDigis");
  me0_pad_token_ = consumes<GEMPadDigiCollection>(me0PadDigis_);
  me0_geom_token_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  config_ = conf;

  // register what this produces
  produces<GE0TriggerDigiCollection>();
}

ME0TriggerProducerV2::~ME0TriggerProducerV2() {}

void ME0TriggerProducerV2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ME0PadDigis", edm::InputTag("me0PadDigis"));
  ME0StubBuilderV2::fillDescription(desc);
  descriptions.add("me0TriggerV2", desc);
}

void ME0TriggerProducerV2::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& setup) const {
  edm::ESHandle<GEMGeometry> h_me0 = setup.getHandle(me0_geom_token_);

  edm::Handle<GEMPadDigiCollection> me0PadDigis;
  ev.getByToken(me0_pad_token_, me0PadDigis);
  const GEMPadDigiCollection* me0Pads = me0PadDigis.product();

  // Create empty collection
  std::unique_ptr<GE0TriggerDigiCollection> oc_trig(new GE0TriggerDigiCollection);

  std::unique_ptr<ME0StubBuilderV2> trigBuilder(new ME0StubBuilderV2(config_));
  trigBuilder->setME0Geometry(&*h_me0);

  // Fill output collections if valid input collection is available.
  trigBuilder->build(me0Pads, *oc_trig);

  // Put collections in event.
  ev.put(std::move(oc_trig));
}

DEFINE_FWK_MODULE(ME0TriggerProducerV2);
