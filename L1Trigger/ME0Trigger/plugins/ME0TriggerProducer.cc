#include "L1Trigger/ME0Trigger/plugins/ME0TriggerProducer.h"
#include "L1Trigger/ME0Trigger/src/ME0TriggerBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigiCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

ME0TriggerProducer::ME0TriggerProducer(const edm::ParameterSet& conf) 
{
  me0PadDigiClusterProducer_ = conf.getParameter<edm::InputTag>("ME0PadDigiClusterProducer");
  me0_pad_token_ = consumes<ME0PadDigiClusterCollection>(me0PadDigiClusterProducer_);
  config_ = conf;

  // register what this produces
  produces<ME0TriggerDigiCollection>();
}

ME0TriggerProducer::~ME0TriggerProducer() 
{
}

void ME0TriggerProducer::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& setup) const
{
  edm::ESHandle<ME0Geometry> h_me0;
  setup.get<MuonGeometryRecord>().get(h_me0);

  edm::Handle<ME0PadDigiClusterCollection> me0PadDigiClusters; 
  ev.getByToken(me0_pad_token_, me0PadDigiClusters);
  const ME0PadDigiClusterCollection *me0Pads = me0PadDigiClusters.product();
  
  // Create empty collection
  std::unique_ptr<ME0TriggerDigiCollection> oc_trig(new ME0TriggerDigiCollection);
  
  std::unique_ptr<ME0TriggerBuilder> trigBuilder( new ME0TriggerBuilder(config_) );
  trigBuilder->setME0Geometry(&*h_me0);

  // Fill output collections if valid input collection is available.
  if (me0PadDigiClusters.isValid()) {   
    trigBuilder->build(me0Pads, *oc_trig);
  }
  
  // Put collections in event.
  ev.put(std::move(oc_trig));
}
