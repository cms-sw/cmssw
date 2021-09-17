#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitProducer.h"

#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ESRecHitProducer::ESRecHitProducer(edm::ParameterSet const& ps)
    : digiToken_(consumes<ESDigiCollection>(ps.getParameter<edm::InputTag>("ESdigiCollection"))),
      rechitCollection_(ps.getParameter<std::string>("ESrechitCollection")),
      worker_{ESRecHitWorkerFactory::get()->create(ps.getParameter<std::string>("algo"), ps, consumesCollector())} {
  produces<ESRecHitCollection>(rechitCollection_);
}

ESRecHitProducer::~ESRecHitProducer() = default;

void ESRecHitProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Handle<ESDigiCollection> digiHandle;
  const ESDigiCollection* digi = nullptr;
  e.getByToken(digiToken_, digiHandle);

  digi = digiHandle.product();
  LogDebug("ESRecHitInfo") << "total # ESdigis: " << digi->size();

  // Create empty output
  auto rec = std::make_unique<ESRecHitCollection>();

  if (digi) {
    rec->reserve(digi->size());

    worker_->set(es);

    // run the algorithm
    for (ESDigiCollection::const_iterator i(digi->begin()); i != digi->end(); i++) {
      worker_->run(i, *rec);
    }
  }

  e.put(std::move(rec), rechitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ESRecHitProducer);
