#include <iostream>

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

class HcalDigisProducerGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HcalDigisProducerGPU(edm::ParameterSet const& ps);
  ~HcalDigisProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  // input product tokens
  edm::EDGetTokenT<HBHEDigiCollection> hbheDigiToken_;
  edm::EDGetTokenT<QIE11DigiCollection> qie11DigiToken_;

  // type aliases
  using HostCollectionf01 =
      hcal::DigiCollection<hcal::Flavor01, hcal::common::VecStoragePolicy<hcal::CUDAHostAllocatorAlias>>;
  using DeviceCollectionf01 = hcal::DigiCollection<hcal::Flavor01, hcal::common::ViewStoragePolicy>;
  using HostCollectionf5 =
      hcal::DigiCollection<hcal::Flavor5, hcal::common::VecStoragePolicy<hcal::CUDAHostAllocatorAlias>>;
  using DeviceCollectionf5 = hcal::DigiCollection<hcal::Flavor5, hcal::common::ViewStoragePolicy>;
  using HostCollectionf3 =
      hcal::DigiCollection<hcal::Flavor3, hcal::common::VecStoragePolicy<hcal::CUDAHostAllocatorAlias>>;
  using DeviceCollectionf3 = hcal::DigiCollection<hcal::Flavor3, hcal::common::ViewStoragePolicy>;

  // output product tokens
  using ProductTypef01 = cms::cuda::Product<DeviceCollectionf01>;
  edm::EDPutTokenT<ProductTypef01> digisF01HEToken_;
  using ProductTypef5 = cms::cuda::Product<DeviceCollectionf5>;
  edm::EDPutTokenT<ProductTypef5> digisF5HBToken_;
  using ProductTypef3 = cms::cuda::Product<DeviceCollectionf3>;
  edm::EDPutTokenT<ProductTypef3> digisF3HBToken_;

  cms::cuda::ContextState cudaState_;

  /*
    hcal::raw::ConfigurationParameters config_;
    // FIXME move this to use raii
    hcal::raw::InputDataCPU inputCPU_;
    hcal::raw::InputDataGPU inputGPU_;
    hcal::raw::OutputDataGPU outputGPU_;
    hcal::raw::ScratchDataGPU scratchGPU_;
    hcal::raw::OutputDataCPU outputCPU_;
    */

  struct ConfigParameters {
    uint32_t maxChannelsF01HE, maxChannelsF5HB, maxChannelsF3HB, nsamplesF01HE, nsamplesF5HB, nsamplesF3HB;
  };
  ConfigParameters config_;

  // tmp on the host
  HostCollectionf01 hf01_;
  HostCollectionf5 hf5_;
  HostCollectionf3 hf3_;

  // device products
  // NOTE: this module owns memory of the product on the device
  DeviceCollectionf01 df01_;
  DeviceCollectionf5 df5_;
  DeviceCollectionf3 df3_;
};

void HcalDigisProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  // FIXME
  desc.add<edm::InputTag>("hbheDigisLabel", edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("qie11DigiLabel", edm::InputTag("hcalDigis"));
  desc.add<std::string>("digisLabelF01HE", std::string{"f01HEDigisGPU"});
  desc.add<std::string>("digisLabelF5HB", std::string{"f5HBDigisGPU"});
  desc.add<std::string>("digisLabelF3HB", std::string{"f3HBDigisGPU"});
  desc.add<uint32_t>("maxChannelsF01HE", 10000u);
  desc.add<uint32_t>("maxChannelsF5HB", 10000u);
  desc.add<uint32_t>("maxChannelsF3HB", 10000u);
  desc.add<uint32_t>("nsamplesF01HE", 8);
  desc.add<uint32_t>("nsamplesF5HB", 8);
  desc.add<uint32_t>("nsamplesF3HB", 8);

  confDesc.addWithDefaultLabel(desc);
}

HcalDigisProducerGPU::HcalDigisProducerGPU(const edm::ParameterSet& ps)
    : hbheDigiToken_{consumes<HBHEDigiCollection>(ps.getParameter<edm::InputTag>("hbheDigisLabel"))},
      qie11DigiToken_{consumes<QIE11DigiCollection>(ps.getParameter<edm::InputTag>("qie11DigiLabel"))},
      digisF01HEToken_{produces<ProductTypef01>(ps.getParameter<std::string>("digisLabelF01HE"))},
      digisF5HBToken_{produces<ProductTypef5>(ps.getParameter<std::string>("digisLabelF5HB"))},
      digisF3HBToken_{produces<ProductTypef3>(ps.getParameter<std::string>("digisLabelF3HB"))} {
  config_.maxChannelsF01HE = ps.getParameter<uint32_t>("maxChannelsF01HE");
  config_.maxChannelsF5HB = ps.getParameter<uint32_t>("maxChannelsF5HB");
  config_.maxChannelsF3HB = ps.getParameter<uint32_t>("maxChannelsF3HB");
  config_.nsamplesF01HE = ps.getParameter<uint32_t>("nsamplesF01HE");
  config_.nsamplesF5HB = ps.getParameter<uint32_t>("nsamplesF5HB");
  config_.nsamplesF3HB = ps.getParameter<uint32_t>("nsamplesF3HB");

  // call CUDA API functions only if CUDA is available
  edm::Service<CUDAService> cs;
  if (cs and cs->enabled()) {
    // allocate on the device
    cudaCheck(cudaMalloc(
        (void**)&df01_.data,
        config_.maxChannelsF01HE * sizeof(uint16_t) * hcal::compute_stride<hcal::Flavor01>(config_.nsamplesF01HE)));
    cudaCheck(cudaMalloc((void**)&df01_.ids, config_.maxChannelsF01HE * sizeof(uint32_t)));

    cudaCheck(cudaMalloc(
        (void**)&df5_.data,
        config_.maxChannelsF5HB * sizeof(uint16_t) * hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB)));
    cudaCheck(cudaMalloc((void**)&df5_.ids, config_.maxChannelsF5HB * sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void**)&df5_.npresamples, sizeof(uint8_t) * config_.maxChannelsF5HB));

    cudaCheck(cudaMalloc(
        (void**)&df3_.data,
        config_.maxChannelsF3HB * sizeof(uint16_t) * hcal::compute_stride<hcal::Flavor3>(config_.nsamplesF3HB)));
    cudaCheck(cudaMalloc((void**)&df3_.ids, config_.maxChannelsF3HB * sizeof(uint32_t)));
  }

  // preallocate on the host
  hf01_.stride = hcal::compute_stride<hcal::Flavor01>(config_.nsamplesF01HE);
  hf5_.stride = hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB);
  hf3_.stride = hcal::compute_stride<hcal::Flavor3>(config_.nsamplesF3HB);
  hf01_.reserve(config_.maxChannelsF01HE);
  hf5_.reserve(config_.maxChannelsF5HB);
  hf3_.reserve(config_.maxChannelsF3HB);
}

HcalDigisProducerGPU::~HcalDigisProducerGPU() {
  // call CUDA API functions only if CUDA is available
  edm::Service<CUDAService> cs;
  if (cs and cs->enabled()) {
    // deallocate on the device
    cudaCheck(cudaFree(df01_.data));
    cudaCheck(cudaFree(df01_.ids));

    cudaCheck(cudaFree(df5_.data));
    cudaCheck(cudaFree(df5_.ids));
    cudaCheck(cudaFree(df5_.npresamples));

    cudaCheck(cudaFree(df3_.data));
    cudaCheck(cudaFree(df3_.ids));
  }
}

void HcalDigisProducerGPU::acquire(edm::Event const& event,
                                   edm::EventSetup const& setup,
                                   edm::WaitingTaskWithArenaHolder holder) {
  // raii
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  hf01_.clear();
  hf5_.clear();
  hf3_.clear();

  // event data
  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<QIE11DigiCollection> qie11Digis;
  event.getByToken(hbheDigiToken_, hbheDigis);
  event.getByToken(qie11DigiToken_, qie11Digis);

  for (auto const& hbhe : *hbheDigis) {
    auto const id = hbhe.id().rawId();
    auto const presamples = hbhe.presamples();
    hf5_.ids.push_back(id);
    hf5_.npresamples.push_back(presamples);
    int stride = hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB);
    // simple for now...
    static_assert(hcal::Flavor5::HEADER_WORDS == 1);
    uint16_t header_word = (1 << 15) | (0x5 << 12) | (0 << 10) | ((hbhe.sample(0).capid() & 0x3) << 8);
    hf5_.data.push_back(header_word);
    //for (int i=0; i<hcal::Flavor5::HEADER_WORDS; i++)
    //    hf5_.data.push_back(0);
    for (int i = 0; i < stride - hcal::Flavor5::HEADER_WORDS; i++) {
      uint16_t s0 = (0 << 7) | (static_cast<uint8_t>(hbhe.sample(2 * i).adc()) & 0x7f);
      uint16_t s1 = (0 << 7) | (static_cast<uint8_t>(hbhe.sample(2 * i + 1).adc()) & 0x7f);
      uint16_t sample = (s1 << 8) | s0;
      hf5_.data.push_back(sample);
    }
  }

  for (unsigned int i = 0; i < qie11Digis->size(); i++) {
    auto const& digi = QIE11DataFrame{(*qie11Digis)[i]};
    if (digi.flavor() == 0 or digi.flavor() == 1) {
      if (digi.detid().subdetId() != HcalEndcap)
        continue;
      auto const id = digi.detid().rawId();
      hf01_.ids.push_back(id);
      for (int hw = 0; hw < hcal::Flavor01::HEADER_WORDS; hw++)
        hf01_.data.push_back((*qie11Digis)[i][hw]);
      for (int sample = 0; sample < digi.samples(); sample++) {
        hf01_.data.push_back((*qie11Digis)[i][hcal::Flavor01::HEADER_WORDS + sample]);
      }
    } else if (digi.flavor() == 3) {
      if (digi.detid().subdetId() != HcalBarrel)
        continue;
      auto const id = digi.detid().rawId();
      hf3_.ids.push_back(id);
      for (int hw = 0; hw < hcal::Flavor3::HEADER_WORDS; hw++)
        hf3_.data.push_back((*qie11Digis)[i][hw]);
      for (int sample = 0; sample < digi.samples(); sample++) {
        hf3_.data.push_back((*qie11Digis)[i][hcal::Flavor3::HEADER_WORDS + sample]);
      }
    }
  }

  auto lambdaToTransfer = [&ctx](auto* dest, auto const& src) {
    using vector_type = typename std::remove_reference<decltype(src)>::type;
    using type = typename vector_type::value_type;
    cudaCheck(cudaMemcpyAsync(dest, src.data(), src.size() * sizeof(type), cudaMemcpyHostToDevice, ctx.stream()));
  };

  lambdaToTransfer(df01_.data, hf01_.data);
  lambdaToTransfer(df01_.ids, hf01_.ids);

  lambdaToTransfer(df5_.data, hf5_.data);
  lambdaToTransfer(df5_.ids, hf5_.ids);
  lambdaToTransfer(df5_.npresamples, hf5_.npresamples);

  lambdaToTransfer(df3_.data, hf3_.data);
  lambdaToTransfer(df3_.ids, hf3_.ids);
}

void HcalDigisProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  df01_.stride = hcal::compute_stride<hcal::Flavor01>(config_.nsamplesF01HE);
  df01_.size = hf01_.ids.size();
  df5_.stride = hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB);
  df5_.size = hf5_.ids.size();
  df3_.stride = hcal::compute_stride<hcal::Flavor3>(config_.nsamplesF3HB);
  df3_.size = hf3_.ids.size();

  ctx.emplace(event, digisF01HEToken_, df01_);
  ctx.emplace(event, digisF5HBToken_, df5_);
  ctx.emplace(event, digisF3HBToken_, df3_);

  /*

#ifdef HCAL_RAWDECODE_CPUDEBUG
    printf("f01he channels = %u f5hb channesl = %u\n",
        outputCPU_.nchannels[hcal::raw::OutputF01HE], 
        outputCPU_.nchannels[hcal::raw::OutputF5HB]);
#endif

    // FIXME: use sizes of views directly for cuda mem cpy?
    auto const nchannelsF01HE = outputCPU_.nchannels[hcal::raw::OutputF01HE];
    auto const nchannelsF5HB = outputCPU_.nchannels[hcal::raw::OutputF5HB];
    outputGPU_.digisF01HE.size = nchannelsF01HE;
    outputGPU_.digisF5HB.size = nchannelsF5HB;
    outputGPU_.digisF01HE.stride = 
        hcal::compute_stride<hcal::Flavor01>(config_.nsamplesF01HE);
    outputGPU_.digisF5HB.stride = 
        hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB);

    hcal::DigiCollection<hcal::Flavor01> digisF01HE{outputGPU_.idsF01HE,
        outputGPU_.digisF01HE, nchannelsF01HE, 
        hcal::compute_stride<hcal::Flavor01>(config_.nsamplesF01HE)};
    hcal::DigiCollection<hcal::Flavor5> digisF5HB{outputGPU_.idsF5HB,
        outputGPU_.digisF5HB, outputGPU_.npresamplesF5HB, nchannelsF5HB, 
        hcal::compute_stride<hcal::Flavor5>(config_.nsamplesF5HB)};

    ctx.emplace(event, digisF01HEToken_, std::move(outputGPU_.digisF01HE));
    ctx.emplace(event, digisF5HBToken_, std::move(outputGPU_.digisF5HB));

    */
}

DEFINE_FWK_MODULE(HcalDigisProducerGPU);
