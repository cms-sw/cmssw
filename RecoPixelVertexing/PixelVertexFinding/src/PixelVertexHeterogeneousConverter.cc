#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"


class PixelVertexHeterogeneousConverter: public edm::global::EDProducer<> {
public:
  explicit PixelVertexHeterogeneousConverter(edm::ParameterSet const& iConfig);
  ~PixelVertexHeterogeneousConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<HeterogeneousProduct> token_;
};

PixelVertexHeterogeneousConverter::PixelVertexHeterogeneousConverter(edm::ParameterSet const& iConfig):
  token_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<reco::VertexCollection>();
}

void PixelVertexHeterogeneousConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pixelVerticesHeterogeneous"));

  descriptions.addWithDefaultLabel(desc);
}

namespace {
  template <typename T>
  auto copy_unique(const T& t) {
    return std::make_unique<T>(t);
  }
}

void PixelVertexHeterogeneousConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HeterogeneousProduct> hinput;
  iEvent.getByToken(token_, hinput);

  const auto& input = hinput->get<pixelVertexHeterogeneousProduct::HeterogeneousPixelVertices>().getProduct<HeterogeneousDevice::kCPU>();

  iEvent.put(copy_unique(input.collection));
}


DEFINE_FWK_MODULE(PixelVertexHeterogeneousConverter);
