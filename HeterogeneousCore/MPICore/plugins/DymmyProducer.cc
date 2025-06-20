#include "DataFormats/Common/interface/FixedSizeDummy.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DummyProducer : public edm::global::EDProducer<> {
public:
  explicit DummyProducer(const edm::ParameterSet& config)
      : sizeInBytes_(config.getParameter<unsigned int>("sizeInBytes")) {
    produces<edm::FixedSizeDummy>();
  }

  void produce(edm::StreamID, edm::Event& event, const edm::EventSetup&) const override {
    auto data = std::make_unique<edm::FixedSizeDummy>(sizeInBytes_);
    event.put(std::move(data));
  }

private:
  unsigned int sizeInBytes_;
};

// Declare as a plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DummyProducer);
