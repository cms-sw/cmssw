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

class EcalPhase2DigiToGPUProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalPhase2DigiToGPUProducer(const edm::ParameterSet& ps);
  ~EcalPhase2DigiToGPUProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder holder) override;
  void produce(edm::Event& evt, edm::EventSetup const& setup) override;

private:
  const edm::EDGetTokenT<EBDigiCollectionPh2> digiCollectionToken_;
  const edm::EDPutTokenT<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>
      digisCollectionToken_;
  uint32_t size_;

  ecal::DigisCollection<::calo::common::DevStoragePolicy> digis_;

  cms::cuda::ContextState cudaState_;
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

void EcalPhase2DigiToGPUProducer::acquire(edm::Event const& event,
                                          edm::EventSetup const& setup,
                                          edm::WaitingTaskWithArenaHolder holder) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  //input data from event
  const auto& pdigis = event.get(digiCollectionToken_);

  size_ = pdigis.size();

  digis_.size = size_;
  //allocate device pointers for output
  digis_.ids = cms::cuda::make_device_unique<uint32_t[]>(size_, ctx.stream());
  digis_.data = cms::cuda::make_device_unique<uint16_t[]>(size_ * EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

  //allocate host pointers for holding product data and id vectors
  auto idstmp = cms::cuda::make_host_unique<uint32_t[]>(size_, ctx.stream());
  auto datatmp = cms::cuda::make_host_unique<uint16_t[]>(size_ * EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

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
      cudaMemcpyAsync(digis_.ids.get(), idstmp.get(), size_ * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(digis_.data.get(),
                            datatmp.get(),
                            size_ * EcalDataFrame_Ph2::MAXSAMPLES * sizeof(uint16_t),
                            cudaMemcpyHostToDevice,
                            ctx.stream()));
}

void EcalPhase2DigiToGPUProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  //get cuda context state for producer
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  //emplace output in the context
  ctx.emplace(event, digisCollectionToken_, std::move(digis_));
}

DEFINE_FWK_MODULE(EcalPhase2DigiToGPUProducer);
