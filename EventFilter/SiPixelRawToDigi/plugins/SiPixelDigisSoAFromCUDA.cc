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
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"

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

  cms::cuda::PortableHostCollection<SiPixelDigisSoALayout<>> digis_h_;

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

  const auto& digis_d = ctx.get(iEvent, digiGetToken_);

  nDigis_ = digis_d.nDigis();
  nDigis_ = digis_d.nDigis();
  digis_h_ = cms::cuda::PortableHostCollection<SiPixelDigisSoALayout<>>(digis_d.view().metadata().size(), ctx.stream());
  cudaCheck(cudaMemcpyAsync(digis_h_.buffer().get(),
                            digis_d.const_buffer().get(),
                            digis_d.bufferSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));
}

void SiPixelDigisSoAFromCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iEvent.emplace(digiPutToken_,
                 nDigis_,
                 digis_h_.view().pdigi(),
                 digis_h_.view().rawIdArr(),
                 digis_h_.view().adc(),
                 digis_h_.view().clus());
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelDigisSoAFromCUDA);
