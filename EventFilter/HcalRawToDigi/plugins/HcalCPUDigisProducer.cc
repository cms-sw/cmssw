#include <iostream>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"

class HcalCPUDigisProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HcalCPUDigisProducer(edm::ParameterSet const& ps);
  ~HcalCPUDigisProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  using IProductTypef01 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor01, hcal::common::ViewStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef01> digisF01HETokenIn_;
  using IProductTypef5 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor5, hcal::common::ViewStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef5> digisF5HBTokenIn_;
  using IProductTypef3 = cms::cuda::Product<hcal::DigiCollection<hcal::Flavor3, hcal::common::ViewStoragePolicy>>;
  edm::EDGetTokenT<IProductTypef3> digisF3HBTokenIn_;

  using OProductTypef01 =
      hcal::DigiCollection<hcal::Flavor01, hcal::common::VecStoragePolicy<hcal::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OProductTypef01> digisF01HETokenOut_;
  using OProductTypef5 =
      hcal::DigiCollection<hcal::Flavor5, hcal::common::VecStoragePolicy<hcal::CUDAHostAllocatorAlias>>;
  edm::EDPutTokenT<OProductTypef5> digisF5HBTokenOut_;
  using OProductTypef3 =
      hcal::DigiCollection<hcal::Flavor3, hcal::common::VecStoragePolicy<hcal::CUDAHostAllocatorAlias>>;
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

  std::string label = "hcalCPUDigisProducer";
  confDesc.add(label, desc);
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

  /*
    idsf01he.resize(f01HEDigis.ndigis);
    dataf01he.resize(f01HEDigis.ndigis * f01HEDigis.stride);
    idsf5hb.resize(f5HBDigis.ndigis);
    npresamplesf5hb.resize(f5HBDigis.ndigis);
    dataf5hb.resize(f5HBDigis.ndigis * f5HBDigis.stride);
    stridef01he = f01HEDigis.stride;
    stridef5hb = f5HBDigis.stride;
    */

  auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using type = typename vector_type::value_type;
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  lambdaToTransfer(digisf01HE_.data, f01HEDigis.data);
  lambdaToTransfer(digisf01HE_.ids, f01HEDigis.ids);

  lambdaToTransfer(digisf5HB_.data, f5HBDigis.data);
  lambdaToTransfer(digisf5HB_.ids, f5HBDigis.ids);
  lambdaToTransfer(digisf5HB_.npresamples, f5HBDigis.npresamples);

  lambdaToTransfer(digisf3HB_.data, f3HBDigis.data);
  lambdaToTransfer(digisf3HB_.ids, f3HBDigis.ids);

  /*
    // enqeue transfers
    cudaCheck( cudaMemcpyAsync(digisf01.data.data(),
                               f01HEDigis.data,
                               dataf01HE.data.size() * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost,
                               ctx.stream().id()) );
    cudaCheck( cudaMemcpyAsync(dataf5hb.data(),
                               f5HBDigis.data,
                               dataf5hb.size() * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost,
                               ctx.stream().id()) );
    cudaCheck( cudaMemcpyAsync(idsf01he.data(),
                               f01HEDigis.ids,
                               idsf01he.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               ctx.stream().id()) );
    cudaCheck( cudaMemcpyAsync(idsf5hb.data(),
                               f5HBDigis.ids,
                               idsf5hb.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               ctx.stream().id()) );
    cudaCheck( cudaMemcpyAsync(npresamplesf5hb.data(),
                               f5HBDigis.npresamples,
                               npresamplesf5hb.size() * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost,
                               ctx.stream.id()) );
                               */
}

void HcalCPUDigisProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  auto outf01 = std::make_unique<OProductTypef01>(std::move(digisf01HE_));
  auto outf5 = std::make_unique<OProductTypef5>(std::move(digisf5HB_));
  auto outf3 = std::make_unique<OProductTypef3>(std::move(digisf3HB_));

  event.put(digisF01HETokenOut_, std::move(outf01));
  event.put(digisF5HBTokenOut_, std::move(outf5));
  event.put(digisF3HBTokenOut_, std::move(outf3));

  // output collections
  /*
    auto f01he = std::make_unique<edm::DataFrameContainer>(
        stridef01he, HcalEndcap, idsf01he.size());
    auto f5hb = std::make_unique<edm::DataFrameContainer>(
        stridef5hb, HcalBarrel, idsf5hb.size());
    
    // cast constness away
    // use pointers to buffers instead of move operator= semantics (or swap)
    // cause we have different allocators in there...
    auto *dataf01hetmp = const_cast<uint16_t*>(f01he->data().data());
    auto *dataf5hbtmp = const_cast<uint16_t*>(f5hb->data().data());

    auto *idsf01hetmp = const_cast<uint32_t*>(f01he->ids().data());
    auto idsf5hbtmp = const_cast<uint32_t*>(f5hb->ids().data());

    // copy data
    std::memcpy(dataf01hetmp, dataf01he.data(), dataf01he.size() * sizeof(uint16_t));
    std::memcpy(dataf5hbtmp, dataf5hb.data(), dataf5hb.size() * sizeof(uint16_t));
    std::memcpy(idsf01hetmp, idsf01he.data(), idsf01he.size() * sizeof(uint32_t));
    std::memcpy(idsf5hbtmp, idsf5hb.data(), idsf5hb.size() * sizeof(uint32_t));

    event.put(digisF01HETokenOut_, std::move(f01he));
    event.put(digisF5HBTokenOut_, std::move(f5hb));
    */
}

DEFINE_FWK_MODULE(HcalCPUDigisProducer);
