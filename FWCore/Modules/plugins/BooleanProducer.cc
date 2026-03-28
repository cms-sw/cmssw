#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class BooleanProducer : public global::EDProducer<> {
  public:
    explicit BooleanProducer(ParameterSet const& config)
        : value_(config.getParameter<bool>("value")), token_(produces<bool>()) {}

    void produce(StreamID sid, Event& event, EventSetup const& setup) const final { event.emplace(token_, value_); }

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    const bool value_;
    const edm::EDPutTokenT<bool> token_;
  };

  void BooleanProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("value", false);
    descriptions.addWithDefaultLabel(desc);
    descriptions.setComment("This EDProducer produces a boolean value according to the \"value\" parameter.");
  }
}  // namespace edm

#include "FWCore/Framework/interface/MakerMacros.h"
using edm::BooleanProducer;
DEFINE_FWK_MODULE(BooleanProducer);
