#include <string>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

class StringProducer : public edm::global::EDProducer<> {

  public:
    explicit StringProducer(edm::ParameterSet const& config);
    ~StringProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    void produce(edm::StreamID, edm::Event & event, edm::EventSetup const& setup) const override;

  private:
    std::string message_;
};

StringProducer::StringProducer(edm::ParameterSet const& config) :
  message_(config.getParameter<std::string>("message"))
{
  produces<std::string>();
}

void StringProducer::produce(edm::StreamID sid, edm::Event & event, edm::EventSetup const& setup) const {
  event.put(std::make_unique<std::string>(message_));
}

void StringProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("message", "");
  descriptions.add(defaultModuleLabel<StringProducer>(), desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StringProducer);
