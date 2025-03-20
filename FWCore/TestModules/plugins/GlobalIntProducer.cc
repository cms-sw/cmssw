#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class GlobalIntProducer : public edm::global::EDProducer<> {
  public:
    explicit GlobalIntProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDPutTokenT<int> token_;
    int value_;
  };

  GlobalIntProducer::GlobalIntProducer(edm::ParameterSet const& config)
      : token_(produces()), value_(config.getParameter<int>("value")) {}

  void GlobalIntProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    event.emplace(token_, value_);
  }

  void GlobalIntProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int>("value", 0);
    descriptions.addDefault(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalIntProducer);
