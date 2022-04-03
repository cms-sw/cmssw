#include <cstdlib>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CUDADataFormats/EcalDigi/interface/DigisCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/ElectronicsMappingGPU.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatiosGPU.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "DeclsForKernelsPh2WeightsGPU.h"
#include "DeclsForKernels.h"

class EcalPh2DigiToGPUProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit EcalPh2DigiToGPUProducer(const edm::ParameterSet& ps);
  ~EcalPh2DigiToGPUProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder holder) override;
  void produce(edm::Event& evt, edm::EventSetup const& setup) override;

private:
  const edm::EDGetTokenT<EBDigiCollectionPh2> ebDigiCollectionToken_;
  const edm::EDPutTokenT<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>
      digisCollectionTokenEB_;
  uint32_t neb_;

  ecal::DigisCollection<::calo::common::DevStoragePolicy> ebdigis_;

  cms::cuda::ContextState cudaState_;
};

void EcalPh2DigiToGPUProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("BarrelDigis", edm::InputTag("simEcalUnsuppressedDigis", ""));
  desc.add<std::string>("digisLabelEB", "ebDigis");

  descriptions.addWithDefaultLabel(desc);
}

EcalPh2DigiToGPUProducer::EcalPh2DigiToGPUProducer(const edm::ParameterSet& ps)
    : ebDigiCollectionToken_(consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("BarrelDigis"))),
      digisCollectionTokenEB_(produces<cms::cuda::Product<ecal::DigisCollection<calo::common::DevStoragePolicy>>>(
          ps.getParameter<std::string>("digisLabelEB"))) {}

void EcalPh2DigiToGPUProducer::acquire(edm::Event const& event,
                                       edm::EventSetup const& setup,
                                       edm::WaitingTaskWithArenaHolder holder) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  //input data from event
  const EBDigiCollectionPh2* pdigis = &event.get(ebDigiCollectionToken_);

  neb_ = pdigis->size();

  //allocate device pointers for output
  ebdigis_.ids = cms::cuda::make_device_unique<uint32_t[]>(neb_, ctx.stream());
  ebdigis_.data = cms::cuda::make_device_unique<uint16_t[]>(neb_ * EcalDataFrame_Ph2::MAXSAMPLES, ctx.stream());

  //allocate host vectors for holding product data and id vectors

  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> idsebtmp;
  std::vector<uint16_t, cms::cuda::HostAllocator<uint16_t>> dataebtmp;

  //resize vectors to number of digis * size needed per digi

  idsebtmp.resize(neb_);
  dataebtmp.resize(neb_ * EcalDataFrame_Ph2::MAXSAMPLES);

  //iterate over digis
  uint32_t i = 0;
  for (auto itdg = pdigis->begin(); itdg != pdigis->end(); ++itdg) {
    //extract digi input
    EcalDataFrame_Ph2 dataFrame(*itdg);

    int nSamples = dataFrame.size();
    //assign id to output vector
    idsebtmp.data()[i] = dataFrame.id();

    //iterate over sample in digi
    for (int sample = 0; sample < nSamples; ++sample) {
      //get samples from input digi
      EcalLiteDTUSample thisSample = dataFrame[sample];
      //assign adc data to output
      dataebtmp.data()[i * nSamples + sample] = thisSample.raw();
    }
    i++;
  }

  //copy output vectors into member variable device pointers for the output struct

  cudaCheck(cudaMemcpyAsync(
      ebdigis_.ids.get(), idsebtmp.data(), idsebtmp.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx.stream()));
  cudaCheck(cudaMemcpyAsync(
      ebdigis_.data.get(), dataebtmp.data(), dataebtmp.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, ctx.stream()));
}

void EcalPh2DigiToGPUProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  //get cuda context state for producer
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  ebdigis_.size = neb_;

  //emplace output in the context
  ctx.emplace(event, digisCollectionTokenEB_, std::move(ebdigis_));
}

DEFINE_FWK_MODULE(EcalPh2DigiToGPUProducer);
