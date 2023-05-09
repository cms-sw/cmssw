/*
 */
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include <memory>

class SiStripSOAtoHost {
public:
  SiStripSOAtoHost() = default;
  void makeAsync(const SiStripClustersCUDADevice& clusters_d, cudaStream_t stream) {
    hostView_ = std::make_unique<SiStripClustersCUDAHost>(clusters_d, stream);
  }
  std::unique_ptr<SiStripClustersCUDAHost> getResults() { return std::move(hostView_); }

private:
  std::unique_ptr<SiStripClustersCUDAHost> hostView_;
};

class SiStripClustersSOAtoHost final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiStripClustersSOAtoHost(const edm::ParameterSet& conf)
      : inputToken_(
            consumes<cms::cuda::Product<SiStripClustersCUDADevice>>(conf.getParameter<edm::InputTag>("ProductLabel"))),
        outputToken_(produces<SiStripClustersCUDAHost>()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add("ProductLabel", edm::InputTag("siStripClusterizerFromRawGPU"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void acquire(edm::Event const& ev,
               edm::EventSetup const& es,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
    const auto& wrapper = ev.get(inputToken_);

    // Sets the current device and creates a CUDA stream
    cms::cuda::ScopedContextAcquire ctx{wrapper, std::move(waitingTaskHolder)};

    const auto& input = ctx.get(wrapper);

    // Queues asynchronous data transfers and kernels to the CUDA stream
    // returned by cms::cuda::ScopedContextAcquire::stream()
    gpuAlgo_.makeAsync(input, ctx.stream());

    // Destructor of ctx queues a callback to the CUDA stream notifying
    // waitingTaskHolder when the queued asynchronous work has finished
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override { ev.put(gpuAlgo_.getResults()); }

private:
  SiStripSOAtoHost gpuAlgo_;

  edm::EDGetTokenT<cms::cuda::Product<SiStripClustersCUDADevice>> inputToken_;
  edm::EDPutTokenT<SiStripClustersCUDAHost> outputToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClustersSOAtoHost);
