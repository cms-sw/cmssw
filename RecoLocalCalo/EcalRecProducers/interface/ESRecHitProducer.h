#ifndef RecoLocalCalo_EcalRecProducers_ESRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_ESRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"

// ESRecHitProducer author : Chia-Ming, Kuo

class ESRecHitProducer : public edm::EDProducer {

 public:

  explicit ESRecHitProducer(const edm::ParameterSet& ps);
  virtual ~ESRecHitProducer();
  virtual void produce(edm::Event& e, const edm::EventSetup& es);

 private:

  edm::InputTag digiCollection_; // secondary name given to collection of digis
  std::string rechitCollection_; // secondary name to be given to collection of hits

  ESRecHitSimAlgo *algo_;

};
#endif
