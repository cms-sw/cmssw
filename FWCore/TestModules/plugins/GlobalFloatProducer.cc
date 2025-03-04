#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class GlobalFloatProducer : public edm::global::EDProducer<> {
  public:
    explicit GlobalFloatProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDPutTokenT<float> token_;
    float value_;
  };

  GlobalFloatProducer::GlobalFloatProducer(edm::ParameterSet const& config)
      : token_(produces()), value_(static_cast<float>(config.getParameter<double>("value"))) {}

  void GlobalFloatProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    event.emplace(token_, value_);
  }

  void GlobalFloatProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<double>("value", 0.);
    descriptions.addDefault(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalFloatProducer);
