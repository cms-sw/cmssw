/*
 */
#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"
#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsGPURcd.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Likely.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "SiStripRawToClusterGPUKernel.h"
#include "ChannelLocsGPU.h"

//#include <sstream>
#include <memory>
#include <mutex>

namespace {
  std::unique_ptr<sistrip::FEDBuffer> fillBuffer(int fedId, const FEDRawData& rawData) {
    std::unique_ptr<sistrip::FEDBuffer> buffer;

    // Check on FEDRawData pointer
    const auto st_buffer = sistrip::preconstructCheckFEDBuffer(rawData);
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
      LogDebug(sistrip::mlRawToCluster_) << "[ClustersFromRawProducer::" << __func__ << "]" << st_buffer
                                         << " for FED ID " << fedId;
      return buffer;
    }
    buffer = std::make_unique<sistrip::FEDBuffer>(rawData);
    const auto st_chan = buffer->findChannels();
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
      LogDebug(sistrip::mlRawToCluster_) << "Exception caught when creating FEDBuffer object for FED " << fedId << ": "
                                         << st_chan;
      buffer.reset();
      return buffer;
    }
    if UNLIKELY (!buffer->doChecks(false)) {
      LogDebug(sistrip::mlRawToCluster_) << "Exception caught when creating FEDBuffer object for FED " << fedId
                                         << ": FED Buffer check fails";
      buffer.reset();
      return buffer;
    }

    return buffer;
  }
}  // namespace

class SiStripClusterizerFromRawGPU final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiStripClusterizerFromRawGPU(const edm::ParameterSet& conf)
      : buffers(1024),
        raw(1024),
        gpuAlgo_(conf.getParameter<edm::ParameterSet>("Clusterizer")),
        inputToken_(consumes(conf.getParameter<edm::InputTag>("ProductLabel"))),
        outputToken_(produces<cms::cuda::Product<SiStripClustersCUDADevice>>()),
        conditionsToken_(esConsumes(edm::ESInputTag{"", conf.getParameter<std::string>("ConditionsLabel")})),
        cpuConditionsToken_(esConsumes(edm::ESInputTag{"", conf.getParameter<std::string>("ConditionsLabel")})) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& ev,
               edm::EventSetup const& es,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
    const auto& conditions = es.getData(conditionsToken_);
    const auto& cpuConditions = es.getData(cpuConditionsToken_);

    // Sets the current device and creates a CUDA stream
    cms::cuda::ScopedContextAcquire ctx{ev.streamID(), std::move(waitingTaskHolder), ctxState_};

    // get raw data
    auto const& rawData = ev.get(inputToken_);
    run(rawData, cpuConditions);

    // Queues asynchronous data transfers and kernels to the CUDA stream
    // returned by cms::cuda::ScopedContextAcquire::stream()
    gpuAlgo_.makeAsync(raw, buffers, conditions, ctx.stream());

    // Destructor of ctx queues a callback to the CUDA stream notifying
    // waitingTaskHolder when the queued asynchronous work has finished
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    cms::cuda::ScopedContextProduce ctx{ctxState_};

    // Now getResult() returns data in GPU memory that is passed to the
    // constructor of OutputData. cms::cuda::ScopedContextProduce::emplace() wraps the
    // OutputData to cms::cuda::Product<OutputData>. cms::cuda::Product<T> stores also
    // the current device and the CUDA stream since those will be needed
    // in the consumer side.
    ctx.emplace(ev, outputToken_, gpuAlgo_.getResults(ctx.stream()));

    for (auto& buf : buffers)
      buf.reset(nullptr);
  }

private:
  void run(const FEDRawDataCollection& rawColl, const SiStripClusterizerConditions& conditions);
  void fill(uint32_t idet, const FEDRawDataCollection& rawColl, const SiStripClusterizerConditions& conditions);

private:
  std::vector<std::unique_ptr<sistrip::FEDBuffer>> buffers;
  std::vector<const FEDRawData*> raw;
  cms::cuda::ContextState ctxState_;

  stripgpu::SiStripRawToClusterGPUKernel gpuAlgo_;

  edm::EDGetTokenT<FEDRawDataCollection> inputToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiStripClustersCUDADevice>> outputToken_;

  edm::ESGetToken<SiStripClusterizerConditionsGPU, SiStripClusterizerConditionsGPURcd> conditionsToken_;
  edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd> cpuConditionsToken_;
};

void SiStripClusterizerFromRawGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("ProductLabel", edm::InputTag("rawDataCollector"));
  desc.add<std::string>("ConditionsLabel", "");

  edm::ParameterSetDescription clusterizer;
  clusterizer.add<std::string>("Algorithm", "ThreeThresholdAlgorithm");
  clusterizer.add("ChannelThreshold", 2.0);
  clusterizer.add("SeedThreshold", 3.0);
  clusterizer.add("ClusterThreshold", 5.0);
  clusterizer.add("MaxSequentialHoles", 0U);
  clusterizer.add("MaxSequentialBad", 1U);
  clusterizer.add("MaxAdjacentBad", 0U);
  clusterizer.add("KeepLargeClusters", false);

  edm::ParameterSetDescription clusterChargeCut;
  clusterChargeCut.add("value", -1.0);

  clusterizer.add("clusterChargeCut", clusterChargeCut);
  desc.add("Clusterizer", clusterizer);

  descriptions.addWithDefaultLabel(desc);
}

void SiStripClusterizerFromRawGPU::run(const FEDRawDataCollection& rawColl,
                                       const SiStripClusterizerConditions& conditions) {
  // loop over good det in cabling
  for (auto idet : conditions.allDetIds()) {
    fill(idet, rawColl, conditions);
  }  // end loop over dets
}

void SiStripClusterizerFromRawGPU::fill(uint32_t idet,
                                        const FEDRawDataCollection& rawColl,
                                        const SiStripClusterizerConditions& conditions) {
  auto const& det = conditions.findDetId(idet);
  if (!det.valid())
    return;

  // Loop over apv-pairs of det
  for (auto const conn : conditions.currentConnection(det)) {
    if UNLIKELY (!conn)
      continue;

    const uint16_t fedId = conn->fedId();

    // If fed id is null or connection is invalid continue
    if UNLIKELY (!fedId || !conn->isConnected()) {
      continue;
    }

    // If Fed hasnt already been initialised, extract data and initialise
    sistrip::FEDBuffer* buffer = buffers[fedId].get();
    if (!buffer) {
      const FEDRawData& rawData = rawColl.FEDData(fedId);
      raw[fedId] = &rawData;
      buffer = fillBuffer(fedId, rawData).release();
      if (!buffer) {
        continue;
      }
      buffers[fedId].reset(buffer);
    }
    assert(buffer);
  }  // end loop over conn
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterizerFromRawGPU);
