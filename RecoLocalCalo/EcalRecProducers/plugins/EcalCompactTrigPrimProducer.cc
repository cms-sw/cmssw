/** \class EcalCompactTrigPrimProducer
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalCompactTrigPrimProducer.h"

EcalCompactTrigPrimProducer::EcalCompactTrigPrimProducer(const edm::ParameterSet& ps)
    : outCollection_(ps.getParameter<std::string>("outColl")) {
  inCollectionToken_ = consumes<EcalTrigPrimDigiCollection>((ps.getParameter<edm::InputTag>("inColl")));
  produces<EcalTrigPrimCompactColl>(outCollection_);
}

void EcalCompactTrigPrimProducer::produce(edm::Event& event, const edm::EventSetup& es) {
  auto outColl = std::make_unique<EcalTrigPrimCompactColl>();
  edm::Handle<EcalTrigPrimDigiCollection> hTPDigis;
  event.getByToken(inCollectionToken_, hTPDigis);

  const EcalTrigPrimDigiCollection* trigPrims = hTPDigis.product();

  for (EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin(); trigPrim != trigPrims->end();
       ++trigPrim) {
    outColl->setValue(
        trigPrim->id().ieta(), trigPrim->id().iphi(), trigPrim->sample(trigPrim->sampleOfInterest()).raw());
  }
  event.put(std::move(outColl), outCollection_);
}
