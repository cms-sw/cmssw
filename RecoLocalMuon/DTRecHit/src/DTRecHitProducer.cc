/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author
 */

#include "DTRecHitProducer.h"

DTRecHitProducer::DTRecHitProducer(const edm::ParameterSet&){}

DTRecHitProducer::~DTRecHitProducer(){}

void DTRecHitProducer::produce(edm::Event& event, const edm::EventSetup&) {
  Handle<DTDigiCollection> digis; 
  event.getByLabel("dtDigis",simHits);

  


  event.put();
}


