/** \class EcalCompactTrigPrimProducer
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Ph. Gras CEA/IRFU Saclay
 *
 **/

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalCompactTrigPrimProducer : public edm::stream::EDProducer<> {
public:
  EcalCompactTrigPrimProducer(const edm::ParameterSet& ps);
  ~EcalCompactTrigPrimProducer() override {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> inCollectionToken_;

  /*
   * output collections
   */
  std::string outCollection_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalCompactTrigPrimProducer);

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
