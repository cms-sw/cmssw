#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include "DeclsForKernelsPhase2.h"

class EcalPhase2DigiToGPUProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalPhase2DigiToGPUProducer(const edm::ParameterSet& ps);
  ~EcalPhase2DigiToGPUProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& evt, edm::EventSetup const& setup) override;

private:
  const edm::EDGetTokenT<EBDigiCollectionPh2> digiCollectionToken_;
  const edm::EDPutTokenT<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>
      digisCollectionToken_;
};

void EcalPhase2DigiToGPUProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("BarrelDigis", edm::InputTag("simEcalUnsuppressedDigis", ""));
  desc.add<std::string>("digisLabelEB", "ebDigis");

  descriptions.addWithDefaultLabel(desc);
}

EcalPhase2DigiToGPUProducer::EcalPhase2DigiToGPUProducer(const edm::ParameterSet& ps)
    : digiCollectionToken_(consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("BarrelDigis"))),
      digisCollectionToken_(produces<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>(
          ps.getParameter<std::string>("digisLabelEB"))) {}

void EcalPhase2DigiToGPUProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{event.streamID()};

  //input data from event
  const auto& pdigis = event.get(digiCollectionToken_);

  const uint32_t size = pdigis.size();

  ecal::DigisCollection<::calo::common::DevStoragePolicy> digis;
  digis.size = size;

  //allocate device pointers for output
  digis.ids = cms::cuda::make_device_unique<uint32_t[]>(size, ctx.stream());
  digis.data = cms::cuda::make_device_unique<uint16_t[]>(size * EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

  //allocate host pointers for holding product data and id vectors
  auto idstmp = cms::cuda::make_host_unique<uint32_t[]>(size, ctx.stream());
  auto datatmp = cms::cuda::make_host_unique<uint16_t[]>(size * EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

  //iterate over digis
  uint32_t i = 0;
  for (const auto& pdigi : pdigis) {
    const int nSamples = pdigi.size();
    //assign id to output vector
    idstmp.get()[i] = pdigi.id();
    //iterate over sample in digi
    for (int sample = 0; sample < nSamples; ++sample) {
      //get samples from input digi
      EcalLiteDTUSample thisSample = pdigi[sample];
      //assign adc data to output
      datatmp.get()[i * nSamples + sample] = thisSample.raw();
    }
    ++i;
  }

  //copy output vectors into member variable device pointers for the output struct
  cudaCheck(
      cudaMemcpyAsync(digis.ids.get(), idstmp.get(), size * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(digis.data.get(),
                            datatmp.get(),
                            size * EcalDataFrame_Ph2::MAXSAMPLES * sizeof(uint16_t),
                            cudaMemcpyHostToDevice,
                            ctx.stream()));

  //emplace output in the context
  ctx.emplace(event, digisCollectionToken_, std::move(digis));
}

DEFINE_FWK_MODULE(EcalPhase2DigiToGPUProducer);
