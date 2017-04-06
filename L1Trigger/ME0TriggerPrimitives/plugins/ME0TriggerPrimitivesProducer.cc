#include "L1Trigger/ME0TriggerPrimitives/plugins/ME0TriggerPrimitivesProducer.h"
#include "L1Trigger/ME0TriggerPrimitives/src/ME0TriggerPrimitivesBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/ME0LCTDigiCollection.h"

ME0TriggerPrimitivesProducer::ME0TriggerPrimitivesProducer(const edm::ParameterSet& conf) 
{
  me0PadDigiProducer_ = conf.getParameter<edm::InputTag>("ME0PadDigiProducer");

  lctBuilder_.reset( new ME0TriggerPrimitivesBuilder(conf) ); // pass on the conf
  
  me0_pad_token_ = consumes<ME0PadDigiCollection>(me0PadDigiProducer_);

  // register what this produces
  produces<ME0LCTDigiCollection>();
  consumes<ME0PadDigiCollection>(me0PadDigiProducer_);
}

ME0TriggerPrimitivesProducer::~ME0TriggerPrimitivesProducer() 
{
}

void ME0TriggerPrimitivesProducer::produce(edm::Event& ev, const edm::EventSetup& setup) 
{
  edm::Handle<ME0PadDigiCollection> me0PadDigis; 
  ev.getByToken(me0_pad_token_, me0PadDigis);
  const ME0PadDigiCollection *me0Pads = me0PadDigis.product();
  
  // Create empty collection
  std::unique_ptr<ME0LCTDigiCollection> oc_lct(new ME0LCTDigiCollection);
  
  // Fill output collections if valid input collection is available.
  if (me0PadDigis.isValid()) {   
    lctBuilder_->build(me0Pads, *oc_lct);
  }
  
  // Put collections in event.
  ev.put(std::move(oc_lct));
}
