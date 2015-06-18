#ifndef RecoLocalCalo_EcalRecProducers_ESRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_ESRecHitProducer_HH

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitProducer.h"
#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitWorker.h"

// ESRecHitProducer author : Chia-Ming, Kuo

class ESDigiCollection;

class ESRecHitProducer : public edm::stream::EDProducer<> {

 public:

  explicit ESRecHitProducer(const edm::ParameterSet& ps);
  virtual ~ESRecHitProducer();
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;

 private:

  const edm::EDGetTokenT<ESDigiCollection> digiToken_;
  const std::string rechitCollection_; // secondary name to be given to collection of hits

  ESRecHitWorkerBaseClass * worker_;
};
#endif
