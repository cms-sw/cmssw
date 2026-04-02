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
  auto SchemaEvolutionProduct =
      std::make_unique<portabletest::SchemaEvolutionHostCollection>(cms::alpakatools::host(), elems);
  auto& view = SchemaEvolutionProduct->view();

  // portabletest::SEEigenVector::Scalar counter = 0;
  for (int i = 0; i < view.metadata().size(); i++) {
    auto element = view[i];
    element.cOneFloat() = static_cast<float>(i) + 0.1f;
    element.cTwoInt() = static_cast<int>(i);
    element.cThreeDouble() = std::sin(static_cast<double>(i)) * 1e3;

    portabletest::SEEigenVector inputVec;

    for (int r = 0; r < portabletest::SEEigenVector::RowsAtCompileTime; ++r) {
      for (int c = 0; c < portabletest::SEEigenVector::ColsAtCompileTime; ++c) {
        inputVec(r, c) = static_cast<portabletest::SEEigenVector::Scalar>(r);
      }
    }
    // element.compact() = portabletest::SiPixelErrorCompact64{1, 2, 3, 4.0};
    element.eOneVector3d() = inputVec;
    element.cFourArray() = portabletest::SEArray{{i, i + 1, i + 2}};
    element.cpixelType() = portabletest::PixelType::s3;
  }
  std::cout << "Sizeof SiPixelErrorCompact64 " << sizeof(portabletest::SiPixelErrorCompact64) << std::endl;

  view.sOneInt() = 42;
  view.sTwoFloat() = 1.0f / 3.0f;
  view.sThreeDouble() = 1.0 / 10.0;
  view.sPixelType() = portabletest::PixelType::s2;

  event.put(std::move(SchemaEvolutionProduct), "SchemaEvolutionProduct");
}
