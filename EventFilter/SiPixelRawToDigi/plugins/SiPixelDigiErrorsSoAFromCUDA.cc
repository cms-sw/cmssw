#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorsSoA.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

class SiPixelDigiErrorsSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelDigiErrorsSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelDigiErrorsSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigiErrorsCUDA>> digiErrorGetToken_;
  edm::EDPutTokenT<SiPixelErrorsSoA> digiErrorPutToken_;

  cms::cuda::host::unique_ptr<SiPixelErrorCompact[]> data_;
  cms::cuda::SimpleVector<SiPixelErrorCompact> error_ = cms::cuda::make_SimpleVector<SiPixelErrorCompact>(0, nullptr);
  const SiPixelFormatterErrors* formatterErrors_ = nullptr;
};

SiPixelDigiErrorsSoAFromCUDA::SiPixelDigiErrorsSoAFromCUDA(const edm::ParameterSet& iConfig)
    : digiErrorGetToken_(
          consumes<cms::cuda::Product<SiPixelDigiErrorsCUDA>>(iConfig.getParameter<edm::InputTag>("src"))),
      digiErrorPutToken_(produces<SiPixelErrorsSoA>()) {}

void SiPixelDigiErrorsSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelDigiErrorsSoAFromCUDA::acquire(const edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Do the transfer in a CUDA stream parallel to the computation CUDA stream
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder)};
  const auto& gpuDigiErrors = ctx.get(iEvent, digiErrorGetToken_);
  formatterErrors_ = &(gpuDigiErrors.formatterErrors());

  if (gpuDigiErrors.nErrorWords() == 0)
    return;

  auto tmp = gpuDigiErrors.dataErrorToHostAsync(ctx.stream());
  error_ = tmp.first;
  data_ = std::move(tmp.second);
}

void SiPixelDigiErrorsSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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
  iEvent.emplace(digiErrorPutToken_, error_.size(), error_.data(), formatterErrors_);
  error_ = cms::cuda::make_SimpleVector<SiPixelErrorCompact>(0, nullptr);
  data_.reset();
  formatterErrors_ = nullptr;
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigiErrorsSoAFromCUDA);
