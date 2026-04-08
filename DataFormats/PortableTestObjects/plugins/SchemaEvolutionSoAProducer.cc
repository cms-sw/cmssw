#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionHostCollection.h"

// This producer was used to create two different ROOT files
// using SoALayouts that resemble the SoAEvolutionZeroLayout but under different class names (SoAEvolutionOneLayout to SoAEvolutionTwoLayout).
// Later SoAEvolutionOneLayout to SoAEvolutionTwoLayout were changes to test if the schema evolution works correctly (see EvolutionOneAnalyzer.cc to EvolutionTwoAnalyzer.cc).

using CollectionVersion = portabletest::HostCollectionEvolutionZero;
constexpr auto productName = "EvolutionZeroProduct";

class SchemaEvolutionSoAProducer : public edm::global::EDProducer<> {
public:
  explicit SchemaEvolutionSoAProducer(const edm::ParameterSet&);
  ~SchemaEvolutionSoAProducer();

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
};

SchemaEvolutionSoAProducer::SchemaEvolutionSoAProducer(const edm::ParameterSet& iConfig) {
  produces<CollectionVersion>(productName);
}

SchemaEvolutionSoAProducer::~SchemaEvolutionSoAProducer() {}

DEFINE_FWK_MODULE(SchemaEvolutionSoAProducer);

void SchemaEvolutionSoAProducer::produce(edm::StreamID iID, edm::Event& event, const edm::EventSetup& iSetup) const {
  std::size_t elems = 23;
  auto product = std::make_unique<CollectionVersion>(cms::alpakatools::host(), elems);
  auto& view = product->view();

  for (int i = 0; i < view.metadata().size(); i++) {
    auto element = view[i];
    element.cFloat() = static_cast<float>(i) + 0.1f;
    element.cInt() = static_cast<int>(i);
    element.cDouble() = std::sin(static_cast<double>(i)) * 1e3;

    element.cEnum() = portabletest::SEEnumType::s2;

    element.eEigenObject() = portabletest::SEEigenObject({{10 * i + 1.1f, -10 * i - 1.2f},
                                                          {10 * i + 2.3f, -10 * i - 2.4f},
                                                          {10 * i + 3.5f, -10 * i - 3.6f},
                                                          {10 * i + 4.7f, -10 * i - 4.8f}});
  }

  view.sInt() = std::numeric_limits<int>::max() - 7;
  view.sFloat() = 1.0f / 3.0f;
  view.sDouble() = 1.0 / 10.0;
  view.sEnum() = portabletest::SEEnumType::s1;

  std::cout << "Running " << __func__ << " with " << view.metadata().size() << " elements" << std::endl;

  event.put(std::move(product), productName);
}
