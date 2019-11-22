/** \class ME0TriggerProducer
 *
 * Takes ME0 pads as input
 * Produces ME0 trigger objects
 *
 * \author Sven Dildick (TAMU).
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TGEM/interface/ME0TriggerBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

class ME0TriggerBuilder;

class ME0TriggerProducer : public edm::global::EDProducer<> {
public:
  explicit ME0TriggerProducer(const edm::ParameterSet&);
  ~ME0TriggerProducer() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::InputTag me0PadDigis_;
  edm::EDGetTokenT<ME0PadDigiCollection> me0_pad_token_;
  edm::ParameterSet config_;
};

ME0TriggerProducer::ME0TriggerProducer(const edm::ParameterSet& conf) {
  me0PadDigis_ = conf.getParameter<edm::InputTag>("ME0PadDigis");
  me0_pad_token_ = consumes<ME0PadDigiCollection>(me0PadDigis_);
  config_ = conf;

  // register what this produces
  produces<ME0TriggerDigiCollection>();
}

ME0TriggerProducer::~ME0TriggerProducer() {}

void ME0TriggerProducer::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& setup) const {
  edm::ESHandle<ME0Geometry> h_me0;
  setup.get<MuonGeometryRecord>().get(h_me0);

  edm::Handle<ME0PadDigiCollection> me0PadDigis;
  ev.getByToken(me0_pad_token_, me0PadDigis);
  const ME0PadDigiCollection* me0Pads = me0PadDigis.product();

  // Create empty collection
  std::unique_ptr<ME0TriggerDigiCollection> oc_trig(new ME0TriggerDigiCollection);

  std::unique_ptr<ME0TriggerBuilder> trigBuilder(new ME0TriggerBuilder(config_));
  trigBuilder->setME0Geometry(&*h_me0);

  // Fill output collections if valid input collection is available.
  trigBuilder->build(me0Pads, *oc_trig);

  // Put collections in event.
  ev.put(std::move(oc_trig));
}

DEFINE_FWK_MODULE(ME0TriggerProducer);
