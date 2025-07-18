#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class GlobalStringProducer : public edm::global::EDProducer<> {
  public:
    explicit GlobalStringProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDPutTokenT<std::string> token_;
    std::string value_;
  };

  GlobalStringProducer::GlobalStringProducer(edm::ParameterSet const& config)
      : token_(produces()), value_(config.getParameter<std::string>("value")) {}

  void GlobalStringProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    event.emplace(token_, value_);
  }

  void GlobalStringProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("value", "");
    descriptions.addDefault(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalStringProducer);
