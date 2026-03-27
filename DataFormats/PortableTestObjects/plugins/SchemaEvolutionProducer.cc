#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionHostCollection.h"

class SchemaEvolutionProducer : public edm::global::EDProducer<> {
public:
  explicit SchemaEvolutionProducer(const edm::ParameterSet&);
  ~SchemaEvolutionProducer();

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
};

SchemaEvolutionProducer::SchemaEvolutionProducer(const edm::ParameterSet& iConfig) {
  produces<portabletest::SchemaEvolutionHostCollection>("SchemaEvolutionProduct");
}

SchemaEvolutionProducer::~SchemaEvolutionProducer() {}

DEFINE_FWK_MODULE(SchemaEvolutionProducer);

void SchemaEvolutionProducer::produce(edm::StreamID iID, edm::Event& event, const edm::EventSetup& iSetup) const {
  std::size_t elems = 16;
  auto SchemaEvolutionProduct = std::make_unique<portabletest::SchemaEvolutionHostCollection>(cms::alpakatools::host(), elems);
  auto& view = SchemaEvolutionProduct->view();

  for (int i = 0; i < view.metadata().size(); i++) {
    auto element = view[i];
    element.cOneFloat() = static_cast<float>(i) + 0.1f;
    element.cTwoInt() = static_cast<int>(i);
    element.cThreeDouble() = std::sin(static_cast<double>(i)) * 1e3;
    element.eOneVector3d() =
        Eigen::Vector3d(static_cast<double>(i) + 0.1, static_cast<double>(i) + 0.2, static_cast<double>(i) + 0.3);
    element.cFourArray() = portabletest::SEArray{{i, i + 1, i + 2}};
  }
  view.sOneInt() = 42;
  view.sTwoFloat() = 1.0f / 3.0f;
  view.sThreeDouble() = 1.0 / 10.0;

  event.put(std::move(SchemaEvolutionProduct), "SchemaEvolutionProduct");
}
