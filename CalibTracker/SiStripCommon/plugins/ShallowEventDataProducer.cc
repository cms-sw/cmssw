#include "CalibTracker/SiStripCommon/interface/ShallowEventDataProducer.h"
#include "FWCore/Framework/interface/Event.h"

ShallowEventDataProducer::ShallowEventDataProducer(const edm::ParameterSet& iConfig) {
  produces <unsigned int> ( "run"   );
  produces <unsigned int> ( "event" );
}

void ShallowEventDataProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<unsigned int >  run   ( new unsigned int(iEvent.id().run()   ) );
  std::auto_ptr<unsigned int >  event ( new unsigned int(iEvent.id().event() ) );

  iEvent.put( run,   "run"   );
  iEvent.put( event, "event" );
}
