#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  class GlobalVectorProducer : public edm::global::EDProducer<> {
  public:
    explicit GlobalVectorProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::EDPutTokenT<std::vector<double>> token_;
    std::vector<double> values_;
  };

  GlobalVectorProducer::GlobalVectorProducer(edm::ParameterSet const& config)
      : token_(produces()), values_(config.getParameter<std::vector<double>>("values")) {}

  void GlobalVectorProducer::produce(edm::StreamID sid, edm::Event& event, edm::EventSetup const& es) const {
    event.emplace(token_, values_);
  }

  void GlobalVectorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<double>>("values", {});
    descriptions.addDefault(desc);
  }

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::GlobalVectorProducer);
