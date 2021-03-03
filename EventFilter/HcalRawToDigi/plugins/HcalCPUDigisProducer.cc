#include <iostream>

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

class HcalCPUDigisProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HcalCPUDigisProducer(edm::ParameterSet const& ps);
  ~HcalCPUDigisProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  using IProductTypef01 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor1, calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef01> digisF01HETokenIn_;
  using IProductTypef5 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor5, calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef5> digisF5HBTokenIn_;
  using IProductTypef3 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor3, calo::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef3> digisF3HBTokenIn_;

  using OProductTypef01 =
      hcal::DigiCollection<hcal::Flavor1, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OProductTypef01> digisF01HETokenOut_;
  using OProductTypef5 =
      hcal::DigiCollection<hcal::Flavor5, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OProductTypef5> digisF5HBTokenOut_;
  using OProductTypef3 =
      hcal::DigiCollection<hcal::Flavor3, calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OProductTypef3> digisF3HBTokenOut_;

  // needed to pass data from acquire to produce
  OProductTypef01 digisf01HE_;
  OProductTypef5 digisf5HB_;
  OProductTypef3 digisf3HB_;
};

void HcalCPUDigisProducer::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digisLabelF01HEIn", edm::InputTag{"hcalRawToDigiGPU", "f01HEDigisGPU"});
  desc.add<edm::InputTag>("digisLabelF5HBIn", edm::InputTag{"hcalRawToDigiGPU", "f5HBDigisGPU"});
  desc.add<edm::InputTag>("digisLabelF3HBIn", edm::InputTag{"hcalRawToDigiGPU", "f3HBDigisGPU"});
  desc.add<std::string>("digisLabelF01HEOut", "f01HEDigis");
  desc.add<std::string>("digisLabelF5HBOut", "f5HBDigis");
  desc.add<std::string>("digisLabelF3HBOut", "f3HBDigis");

  confDesc.addWithDefaultLabel(desc);
}

HcalCPUDigisProducer::HcalCPUDigisProducer(const edm::ParameterSet& ps)
    : digisF01HETokenIn_{consumes<IProductTypef01>(ps.getParameter<edm::InputTag>("digisLabelF01HEIn"))},
      digisF5HBTokenIn_{consumes<IProductTypef5>(ps.getParameter<edm::InputTag>("digisLabelF5HBIn"))},
      digisF3HBTokenIn_{consumes<IProductTypef3>(ps.getParameter<edm::InputTag>("digisLabelF3HBIn"))},
      digisF01HETokenOut_{produces<OProductTypef01>(ps.getParameter<std::string>("digisLabelF01HEOut"))},
      digisF5HBTokenOut_{produces<OProductTypef5>(ps.getParameter<std::string>("digisLabelF5HBOut"))},
      digisF3HBTokenOut_{produces<OProductTypef3>(ps.getParameter<std::string>("digisLabelF3HBOut"))} {}

HcalCPUDigisProducer::~HcalCPUDigisProducer() {}

void HcalCPUDigisProducer::acquire(edm::Event const& event,
                                   edm::EventSetup const& setup,
                                   edm::WaitingTaskWithArenaHolder taskHolder) {
  // retrieve data/ctx
  auto const& f01HEProduct = event.get(digisF01HETokenIn_);
  auto const& f5HBProduct = event.get(digisF5HBTokenIn_);
  auto const& f3HBProduct = event.get(digisF3HBTokenIn_);
  cms::cuda::ScopedContextAcquire ctx{f01HEProduct, std::move(taskHolder)};
  auto const& f01HEDigis = ctx.get(f01HEProduct);
  auto const& f5HBDigis = ctx.get(f5HBProduct);
  auto const& f3HBDigis = ctx.get(f3HBProduct);

  // resize out tmp buffers
  digisf01HE_.stride = f01HEDigis.stride;
  digisf5HB_.stride = f5HBDigis.stride;
  digisf3HB_.stride = f3HBDigis.stride;
  digisf01HE_.resize(f01HEDigis.size);
  digisf5HB_.resize(f5HBDigis.size);
  digisf3HB_.resize(f3HBDigis.size);

  auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using type = typename vector_type::value_type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  lambdaToTransfer(digisf01HE_.data, f01HEDigis.data.get());
  lambdaToTransfer(digisf01HE_.ids, f01HEDigis.ids.get());

  lambdaToTransfer(digisf5HB_.data, f5HBDigis.data.get());
  lambdaToTransfer(digisf5HB_.ids, f5HBDigis.ids.get());
  lambdaToTransfer(digisf5HB_.npresamples, f5HBDigis.npresamples.get());

  lambdaToTransfer(digisf3HB_.data, f3HBDigis.data.get());
  lambdaToTransfer(digisf3HB_.ids, f3HBDigis.ids.get());
}

void HcalCPUDigisProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  event.emplace(digisF01HETokenOut_, std::move(digisf01HE_));
  event.emplace(digisF5HBTokenOut_, std::move(digisf5HB_));
  event.emplace(digisF3HBTokenOut_, std::move(digisf3HB_));
}

DEFINE_FWK_MODULE(HcalCPUDigisProducer);
