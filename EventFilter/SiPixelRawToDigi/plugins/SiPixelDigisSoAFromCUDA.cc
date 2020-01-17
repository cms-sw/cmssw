#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

class SiPixelDigisSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelDigisSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelDigisSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  cms::cuda::host::unique_ptr<uint32_t[]> pdigi_;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArr_;
  cms::cuda::host::unique_ptr<uint16_t[]> adc_;
  cms::cuda::host::unique_ptr<int32_t[]> clus_;

  int nDigis_;
};

SiPixelDigisSoAFromCUDA::SiPixelDigisSoAFromCUDA(const edm::ParameterSet& iConfig)
    : digiGetToken_(consumes<cms::cuda::Product<SiPixelDigisCUDA>>(iConfig.getParameter<edm::InputTag>("src"))),
      digiPutToken_(produces<SiPixelDigisSoA>()) {}

void SiPixelDigisSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigisSoAFromCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  const auto& gpuDigis = ctx.get(iEvent, digiGetToken_);

  nDigis_ = gpuDigis.nDigis();
  pdigi_ = gpuDigis.pdigiToHostAsync(ctx.stream());
  rawIdArr_ = gpuDigis.rawIdArrToHostAsync(ctx.stream());
  adc_ = gpuDigis.adcToHostAsync(ctx.stream());
  clus_ = gpuDigis.clusToHostAsync(ctx.stream());
}

void SiPixelDigisSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // The following line copies the data from the pinned host memory to
  // regular host memory. In principle that feels unnecessary (why not
  // just use the pinned host memory?). There are a few arguments for
  // doing it though
  // - Now can release the pinned host memory back to the (caching) allocator
  //   * if we'd like to keep the pinned memory, we'd need to also
  //     keep the CUDA stream around as long as that, or allow pinned
  //     host memory to be allocated without a CUDA stream
  // - What if a CPU algorithm would produce the same SoA? We can't
  //   use cudaMallocHost without a GPU...
  iEvent.emplace(digiPutToken_, nDigis_, pdigi_.get(), rawIdArr_.get(), adc_.get(), clus_.get());

  pdigi_.reset();
  rawIdArr_.reset();
  adc_.reset();
  clus_.reset();
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
