#ifndef GeneratorInterface_Core_GeneratorSmearedProducer_h
#define GeneratorInterface_Core_GeneratorSmearedProducer_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;
  class HepMCProduct;
}

class GeneratorSmearedProducer : public edm::global::EDProducer<> {

public:

  explicit GeneratorSmearedProducer(edm::ParameterSet const& p);

  virtual void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<edm::HepMCProduct> newToken_;
  const edm::EDGetTokenT<edm::HepMCProduct> oldToken_;
};

#endif
