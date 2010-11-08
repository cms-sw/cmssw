/** \class EcalCompactTrigPrimProducer
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalCompactTrigPrimProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"


EcalCompactTrigPrimProducer::EcalCompactTrigPrimProducer(const edm::ParameterSet& ps):
  inCollection_(ps.getParameter<edm::InputTag>("inColl")),
  outCollection_(ps.getParameter<std::string>("outColl"))
{

  produces<EcalTrigPrimCompactColl>(outCollection_);
}

void EcalCompactTrigPrimProducer::produce(edm::Event& event, const edm::EventSetup& es)
{
  std::auto_ptr<EcalTrigPrimCompactColl> outColl(new EcalTrigPrimCompactColl);
  edm::Handle<EcalTrigPrimDigiCollection> hTPDigis;
  event.getByLabel(inCollection_, hTPDigis);
  
  const EcalTrigPrimDigiCollection* trigPrims =  hTPDigis.product();
  
  for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin();
      trigPrim != trigPrims->end(); ++trigPrim){
    outColl->setValue(trigPrim->id().ieta(), trigPrim->id().iphi(), trigPrim->sample(trigPrim->sampleOfInterest()).raw());
  }
  event.put(outColl, outCollection_);
}
