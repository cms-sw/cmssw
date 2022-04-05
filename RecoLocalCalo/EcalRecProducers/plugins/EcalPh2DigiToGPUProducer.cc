#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DeclsForKernelsPh2WeightsGPU.h"

class EcalPh2DigiToGPUProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalPh2DigiToGPUProducer(const edm::ParameterSet& ps);
  ~EcalPh2DigiToGPUProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder holder) override;
  void produce(edm::Event& evt, edm::EventSetup const& setup) override;

private:
  const edm::EDGetTokenT<EBDigiCollectionPh2> digiCollectionToken_;
  const edm::EDPutTokenT<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>
      digisCollectionToken_;
  uint32_t n_;

  ecal::DigisCollection<::calo::common::DevStoragePolicy> digis_;

  cms::cuda::ContextState cudaState_;
};

void EcalPh2DigiToGPUProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("BarrelDigis", edm::InputTag("simEcalUnsuppressedDigis", ""));
  desc.add<std::string>("digisLabelEB", "ebDigis");

  descriptions.addWithDefaultLabel(desc);
}

EcalPh2DigiToGPUProducer::EcalPh2DigiToGPUProducer(const edm::ParameterSet& ps)
    : digiCollectionToken_(consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("BarrelDigis"))),
      digisCollectionToken_(produces<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>(
          ps.getParameter<std::string>("digisLabelEB"))) {}

void EcalPh2DigiToGPUProducer::acquire(edm::Event const& event,
                                       edm::EventSetup const& setup,
                                       edm::WaitingTaskWithArenaHolder holder) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  //input data from event
  const auto& pdigis = event.get(digiCollectionToken_);

  n_ = pdigis.size();

  //allocate device pointers for output
  digis_.ids = cms::cuda::make_device_unique<uint32_t[]>(n_, ctx.stream());
  digis_.data = cms::cuda::make_device_unique<uint16_t[]>(n_ * EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

  //allocate host vectors for holding product data and id vectors

  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> idstmp;
  std::vector<uint16_t, cms::cuda::HostAllocator<uint16_t>> datatmp;

  //resize vectors to number of digis * size needed per digi

  idstmp.resize(n_);
  datatmp.resize(n_ * EcalDataFrame_Ph2::MAXSAMPLES);

  //iterate over digis
  uint32_t i = 0;
  for (const auto& pdigi : pdigis) {
    const int nSamples = pdigi.size();
    //assign id to output vector
    idstmp.data()[i] = pdigi.id();
    //iterate over sample in digi
    for (int sample = 0; sample < nSamples; ++sample) {
      //get samples from input digi
      EcalLiteDTUSample thisSample = pdigi[sample];
      //assign adc data to output
      datatmp.data()[i * nSamples + sample] = thisSample.raw();
    }
    ++i;
  }

  //copy output vectors into member variable device pointers for the output struct

  cudaCheck(cudaMemcpyAsync(
      digis_.ids.get(), idstmp.data(), idstmp.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      digis_.data.get(), datatmp.data(), datatmp.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, ctx.stream()));
}

void EcalPh2DigiToGPUProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  //get cuda context state for producer
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  digis_.size = n_;

  //emplace output in the context
  ctx.emplace(event, digisCollectionToken_, std::move(digis_));
}

DEFINE_FWK_MODULE(EcalPh2DigiToGPUProducer);
