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
  ~HcalDigisProducerGPU() override = default;
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
      hcal::DigiCollection<hcal::Flavor1, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using DeviceCollectionf01 = hcal::DigiCollection<hcal::Flavor1, calo::common::DevStoragePolicy>;
  using HostCollectionf5 =
      hcal::DigiCollection<hcal::Flavor5, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using DeviceCollectionf5 = hcal::DigiCollection<hcal::Flavor5, calo::common::DevStoragePolicy>;
  using HostCollectionf3 =
      hcal::DigiCollection<hcal::Flavor3, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using DeviceCollectionf3 = hcal::DigiCollection<hcal::Flavor3, calo::common::DevStoragePolicy>;

  // output product tokens
  using ProductTypef01 = cms::cuda::Product<DeviceCollectionf01>;
  edm::EDPutTokenT<ProductTypef01> digisF01HEToken_;
  using ProductTypef5 = cms::cuda::Product<DeviceCollectionf5>;
  edm::EDPutTokenT<ProductTypef5> digisF5HBToken_;
  using ProductTypef3 = cms::cuda::Product<DeviceCollectionf3>;
  edm::EDPutTokenT<ProductTypef3> digisF3HBToken_;

  cms::cuda::ContextState cudaState_;

  struct ConfigParameters {
    uint32_t maxChannelsF01HE, maxChannelsF5HB, maxChannelsF3HB;
  };
  ConfigParameters config_;

  // per event host buffers
  HostCollectionf01 hf01_;
  HostCollectionf5 hf5_;
  HostCollectionf3 hf3_;

  // device products: product owns memory (i.e. not the module)
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

  // this is a preallocation for the max statically known number of time samples
  // actual stride/nsamples will be inferred from data
  hf01_.stride = hcal::compute_stride<hcal::Flavor1>(QIE11DigiCollection::MAXSAMPLES);
  hf5_.stride = hcal::compute_stride<hcal::Flavor5>(HBHEDataFrame::MAXSAMPLES);
  hf3_.stride = hcal::compute_stride<hcal::Flavor3>(QIE11DigiCollection::MAXSAMPLES);

  // preallocate pinned host memory only if CUDA is available
  edm::Service<CUDAService> cs;
  if (cs and cs->enabled()) {
    hf01_.reserve(config_.maxChannelsF01HE);
    hf5_.reserve(config_.maxChannelsF5HB);
    hf3_.reserve(config_.maxChannelsF3HB);
  }
}

void HcalDigisProducerGPU::acquire(edm::Event const& event,
                                   edm::EventSetup const& setup,
                                   edm::WaitingTaskWithArenaHolder holder) {
  // raii
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

  // clear host buffers
  hf01_.clear();
  hf5_.clear();
  hf3_.clear();

  // event data
  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<QIE11DigiCollection> qie11Digis;
  event.getByToken(hbheDigiToken_, hbheDigis);
  event.getByToken(qie11DigiToken_, qie11Digis);

  // init f5 collection
  if (not hbheDigis->empty()) {
    auto const nsamples = (*hbheDigis)[0].size();
    auto const stride = hcal::compute_stride<hcal::Flavor5>(nsamples);
    hf5_.stride = stride;

    // flavor5 get device blobs
    df5_.stride = stride;
    df5_.data = cms::cuda::make_device_unique<uint16_t[]>(config_.maxChannelsF5HB * stride, ctx.stream());
    df5_.ids = cms::cuda::make_device_unique<uint32_t[]>(config_.maxChannelsF5HB, ctx.stream());
    df5_.npresamples = cms::cuda::make_device_unique<uint8_t[]>(config_.maxChannelsF5HB, ctx.stream());
  }

  if (not qie11Digis->empty()) {
    auto const nsamples = qie11Digis->samples();
    auto const stride01 = hcal::compute_stride<hcal::Flavor1>(nsamples);
    auto const stride3 = hcal::compute_stride<hcal::Flavor3>(nsamples);

    hf01_.stride = stride01;
    hf3_.stride = stride3;

    // flavor 0/1 get devie blobs
    df01_.stride = stride01;
    df01_.data = cms::cuda::make_device_unique<uint16_t[]>(config_.maxChannelsF01HE * stride01, ctx.stream());
    df01_.ids = cms::cuda::make_device_unique<uint32_t[]>(config_.maxChannelsF01HE, ctx.stream());

    // flavor3 get device blobs
    df3_.stride = stride3;
    df3_.data = cms::cuda::make_device_unique<uint16_t[]>(config_.maxChannelsF3HB * stride3, ctx.stream());
    df3_.ids = cms::cuda::make_device_unique<uint32_t[]>(config_.maxChannelsF3HB, ctx.stream());
  }

  for (auto const& hbhe : *hbheDigis) {
    auto const id = hbhe.id().rawId();
    auto const presamples = hbhe.presamples();
    hf5_.ids.push_back(id);
    hf5_.npresamples.push_back(presamples);
    auto const stride = hcal::compute_stride<hcal::Flavor5>(hbhe.size());
    assert(stride == hf5_.stride && "strides must be the same for every single digi of the collection");
    // simple for now...
    static_assert(hcal::Flavor5::HEADER_WORDS == 1);
    uint16_t header_word = (1 << 15) | (0x5 << 12) | (0 << 10) | ((hbhe.sample(0).capid() & 0x3) << 8);
    hf5_.data.push_back(header_word);
    for (unsigned int i = 0; i < stride - hcal::Flavor5::HEADER_WORDS; i++) {
      uint16_t s0 = (0 << 7) | (static_cast<uint8_t>(hbhe.sample(2 * i).adc()) & 0x7f);
      uint16_t s1 = (0 << 7) | (static_cast<uint8_t>(hbhe.sample(2 * i + 1).adc()) & 0x7f);
      uint16_t sample = (s1 << 8) | s0;
      hf5_.data.push_back(sample);
    }
  }

  for (unsigned int i = 0; i < qie11Digis->size(); i++) {
    auto const& digi = QIE11DataFrame{(*qie11Digis)[i]};
    assert(digi.samples() == qie11Digis->samples() && "collection nsamples must equal per digi samples");
    if (digi.flavor() == 0 or digi.flavor() == 1) {
      if (digi.detid().subdetId() != HcalEndcap)
        continue;
      auto const id = digi.detid().rawId();
      hf01_.ids.push_back(id);
      for (int hw = 0; hw < hcal::Flavor1::HEADER_WORDS; hw++)
        hf01_.data.push_back((*qie11Digis)[i][hw]);
      for (int sample = 0; sample < digi.samples(); sample++) {
        hf01_.data.push_back((*qie11Digis)[i][hcal::Flavor1::HEADER_WORDS + sample]);
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
    if (src.empty())
      return;
    using vector_type = typename std::remove_reference<decltype(src)>::type;
    using type = typename vector_type::value_type;
    using dest_data_type = typename std::remove_pointer<decltype(dest)>::type;
    static_assert(std::is_same<dest_data_type, type>::value && "Dest and Src data typesdo not match");
    cudaCheck(cudaMemcpyAsync(dest, src.data(), src.size() * sizeof(type), cudaMemcpyHostToDevice, ctx.stream()));
  };

  lambdaToTransfer(df01_.data.get(), hf01_.data);
  lambdaToTransfer(df01_.ids.get(), hf01_.ids);

  lambdaToTransfer(df5_.data.get(), hf5_.data);
  lambdaToTransfer(df5_.ids.get(), hf5_.ids);
  lambdaToTransfer(df5_.npresamples.get(), hf5_.npresamples);

  lambdaToTransfer(df3_.data.get(), hf3_.data);
  lambdaToTransfer(df3_.ids.get(), hf3_.ids);

  df01_.size = hf01_.ids.size();
  df5_.size = hf5_.ids.size();
  df3_.size = hf3_.ids.size();
}

void HcalDigisProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};

  ctx.emplace(event, digisF01HEToken_, std::move(df01_));
  ctx.emplace(event, digisF5HBToken_, std::move(df5_));
  ctx.emplace(event, digisF3HBToken_, std::move(df3_));
}

DEFINE_FWK_MODULE(HcalDigisProducerGPU);
