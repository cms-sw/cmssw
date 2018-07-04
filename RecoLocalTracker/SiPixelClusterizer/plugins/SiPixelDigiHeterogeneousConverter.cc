#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "siPixelRawToClusterHeterogeneousProduct.h"

class SiPixelDigiHeterogeneousConverter: public edm::global::EDProducer<> {
public:
  explicit SiPixelDigiHeterogeneousConverter(edm::ParameterSet const& iConfig);
  ~SiPixelDigiHeterogeneousConverter() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<HeterogeneousProduct> token_;
  bool includeErrors_;
};

SiPixelDigiHeterogeneousConverter::SiPixelDigiHeterogeneousConverter(edm::ParameterSet const& iConfig):
  token_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src"))),
  includeErrors_(iConfig.getParameter<bool>("includeErrors"))
{
  produces<edm::DetSetVector<PixelDigi> >();
  if(includeErrors_) {
    produces< edm::DetSetVector<SiPixelRawDataError> >();
    produces<DetIdCollection>();
    produces<DetIdCollection>("UserErrorModules");
    produces<edmNew::DetSetVector<PixelFEDChannel> >();
  }
}

void SiPixelDigiHeterogeneousConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersHeterogeneous"));
  desc.add<bool>("includeErrors",true);

  descriptions.addWithDefaultLabel(desc);
}

namespace {
  template <typename T>
  auto copy_unique(const T& t) {
    return std::make_unique<T>(t);
  }
}

void SiPixelDigiHeterogeneousConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HeterogeneousProduct> hinput;
  iEvent.getByToken(token_, hinput);

  const auto& input = hinput->get<siPixelRawToClusterHeterogeneousProduct::HeterogeneousDigiCluster>().getProduct<HeterogeneousDevice::kCPU>();

  iEvent.put(copy_unique(input.collection));
  if(includeErrors_) {
    iEvent.put(copy_unique(input.errorcollection));
    iEvent.put(copy_unique(input.tkerror_detidcollection));
    iEvent.put(copy_unique(input.usererror_detidcollection), "UserErrorModules");
    iEvent.put(copy_unique(input.disabled_channelcollection));
  }
}


DEFINE_FWK_MODULE(SiPixelDigiHeterogeneousConverter);
