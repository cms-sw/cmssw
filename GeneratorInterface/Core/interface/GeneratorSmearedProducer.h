#ifndef GeneratorInterface_Core_GeneratorSmearedProducer_h
#define GeneratorInterface_Core_GeneratorSmearedProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;
  class HepMCProduct;
}

class GeneratorSmearedProducer : public edm::EDProducer {

public:

  explicit GeneratorSmearedProducer(edm::ParameterSet const& p);
  virtual ~GeneratorSmearedProducer() {}
  virtual void produce(edm::Event& e, edm::EventSetup const& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::HepMCProduct> newToken_;
  edm::EDGetTokenT<edm::HepMCProduct> oldToken_;
};

#endif
