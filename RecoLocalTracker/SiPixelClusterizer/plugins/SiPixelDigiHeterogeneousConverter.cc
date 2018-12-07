#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

/**
 * This is very stupid but currently the easiest way to go forward as
 * one can't replace and EDProducer with an EDAlias in the
 * configuration...
 */

class SiPixelDigiHeterogeneousConverter: public edm::global::EDProducer<> {
public:
  explicit SiPixelDigiHeterogeneousConverter(edm::ParameterSet const& iConfig);
  ~SiPixelDigiHeterogeneousConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > token_collection_;
  edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError>> token_errorcollection_;
  edm::EDGetTokenT<DetIdCollection> token_tkerror_detidcollection_;
  edm::EDGetTokenT<DetIdCollection> token_usererror_detidcollection_;
  edm::EDGetTokenT<edmNew::DetSetVector<PixelFEDChannel>> token_disabled_channelcollection_;
  bool includeErrors_;
};

SiPixelDigiHeterogeneousConverter::SiPixelDigiHeterogeneousConverter(edm::ParameterSet const& iConfig):
  includeErrors_(iConfig.getParameter<bool>("includeErrors"))
{
  auto src = iConfig.getParameter<edm::InputTag>("src");

  token_collection_ = consumes<edm::DetSetVector<PixelDigi> >(src);
  produces<edm::DetSetVector<PixelDigi> >();
  if(includeErrors_) {
    token_errorcollection_ = consumes<edm::DetSetVector<SiPixelRawDataError>>(src);
    produces< edm::DetSetVector<SiPixelRawDataError> >();

    token_tkerror_detidcollection_ = consumes<DetIdCollection>(src);
    produces<DetIdCollection>();

    token_usererror_detidcollection_ = consumes<DetIdCollection>(edm::InputTag(src.label(), "UserErrorModules"));
    produces<DetIdCollection>("UserErrorModules");

    token_disabled_channelcollection_ = consumes<edmNew::DetSetVector<PixelFEDChannel>>(src);
    produces<edmNew::DetSetVector<PixelFEDChannel> >();
  }
}

void SiPixelDigiHeterogeneousConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<bool>("includeErrors",true);

  descriptions.addWithDefaultLabel(desc);
}

namespace {
  template <typename T>
  void copy(edm::Event& iEvent, const edm::EDGetTokenT<T>& token) {
    edm::Handle<T> h;
    iEvent.getByToken(token, h);
    iEvent.put(std::make_unique<T>(*h));
  }

  template <typename T>
  void copy(edm::Event& iEvent, const edm::EDGetTokenT<T>& token, const std::string& instance) {
    edm::Handle<T> h;
    iEvent.getByToken(token, h);
    iEvent.put(std::make_unique<T>(*h), instance);
  }
}

void SiPixelDigiHeterogeneousConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  copy(iEvent, token_collection_);
  if(includeErrors_) {
    copy(iEvent, token_errorcollection_);
    copy(iEvent, token_tkerror_detidcollection_);
    copy(iEvent, token_usererror_detidcollection_, "UserErrorModules");
    copy(iEvent, token_disabled_channelcollection_);
  }
}


DEFINE_FWK_MODULE(SiPixelDigiHeterogeneousConverter);
