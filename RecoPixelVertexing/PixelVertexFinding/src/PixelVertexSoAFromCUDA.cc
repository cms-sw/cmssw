#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"

#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

class PixelVertexSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelVertexSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelVertexSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<CUDAProduct<ZVertexHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexHeterogeneous> tokenSOA_;

  cudautils::host::unique_ptr<ZVertexSoA> m_soa;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes<CUDAProduct<ZVertexHeterogeneous>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenSOA_(produces<ZVertexHeterogeneous>()) {}

void PixelVertexSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelVertexCUDA"));
  descriptions.add("pixelVertexSoA", desc);
}

void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  auto const& inputDataWrapped = iEvent.get(tokenCUDA_);
  CUDAScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());
}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  iEvent.emplace(tokenSOA_, ZVertexHeterogeneous(std::move(m_soa)));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
