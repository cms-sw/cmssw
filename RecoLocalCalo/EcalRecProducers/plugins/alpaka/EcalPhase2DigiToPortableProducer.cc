#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiPhase2HostCollection.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EcalPhase2DigiToPortableProducer : public stream::EDProducer<> {
  public:
    explicit EcalPhase2DigiToPortableProducer(edm::ParameterSet const &ps);
    ~EcalPhase2DigiToPortableProducer() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    void produce(device::Event &event, device::EventSetup const &setup) override;

  private:
    const edm::EDGetTokenT<EBDigiCollectionPh2> inputDigiToken_;
    const device::EDPutToken<EcalDigiPhase2DeviceCollection> outputDigiDevToken_;
  };

  void EcalPhase2DigiToPortableProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("BarrelDigis", edm::InputTag("simEcalUnsuppressedDigis", ""));
    desc.add<std::string>("digisLabelEB", "ebDigis");

    descriptions.addWithDefaultLabel(desc);
  }

  EcalPhase2DigiToPortableProducer::EcalPhase2DigiToPortableProducer(edm::ParameterSet const &ps)
      : inputDigiToken_(consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("BarrelDigis"))),
        outputDigiDevToken_(produces(ps.getParameter<std::string>("digisLabelEB"))) {}

  void EcalPhase2DigiToPortableProducer::produce(device::Event &event, device::EventSetup const &setup) {
    //input data from event
    const auto &inputDigis = event.get(inputDigiToken_);

    const uint32_t size = inputDigis.size();

    //create host and device Digi collections of required size
    EcalDigiPhase2DeviceCollection digisDevColl{static_cast<int32_t>(size), event.queue()};
    EcalDigiPhase2HostCollection digisHostColl{static_cast<int32_t>(size), event.queue()};
    auto digisHostCollView = digisHostColl.view();

    //iterate over digis
    uint32_t i = 0;
    for (const auto &inputDigi : inputDigis) {
      const int nSamples = inputDigi.size();
      //assign id to host collection
      digisHostCollView.id()[i] = inputDigi.id();
      //iterate over sample in digi
      for (int sample = 0; sample < nSamples; ++sample) {
        //get samples from input digi
        EcalLiteDTUSample thisSample = inputDigi[sample];
        //assign adc data to host collection
        digisHostCollView.data()[i][sample] = thisSample.raw();
      }
      ++i;
    }
    digisHostCollView.size() = i;

    //copy collection from host to device
    alpaka::memcpy(event.queue(), digisDevColl.buffer(), digisHostColl.buffer());

    //emplace device collection in the event
    event.emplace(outputDigiDevToken_, std::move(digisDevColl));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(EcalPhase2DigiToPortableProducer);
