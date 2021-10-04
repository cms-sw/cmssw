#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

SiStripClusterizerConditionsGPU::SiStripClusterizerConditionsGPU(const SiStripQuality& quality,
                                                                 const SiStripGain* gains,
                                                                 const SiStripNoises& noises) {
  cudaCheck(cudaMallocHost(&conditions_, sizeof(Data)));
  detToFeds_.clear();

  // connected: map<DetID, std::vector<int>>
  // map of KEY=detid DATA=vector of apvs, maximum 6 APVs per detector module :
  const auto& connected = quality.cabling()->connected();
  // detCabling: map<DetID, std::vector<const FedChannelConnection *>
  // map of KEY=detid DATA=vector<FedChannelConnection>
  const auto& detCabling = quality.cabling()->getDetCabling();

  for (const auto& conn : connected) {
    const auto det = conn.first;
    if (!quality.IsModuleBad(det)) {
      const auto detConn_it = detCabling.find(det);

      if (detCabling.end() != detConn_it) {
        for (const auto& chan : (*detConn_it).second) {
          if (chan && chan->fedId() && chan->isConnected()) {
            const auto detID = chan->detId();
            const auto fedID = chan->fedId();
            const auto fedCh = chan->fedCh();
            const auto iPair = chan->apvPairNumber();

            detToFeds_.emplace_back(detID, iPair, fedID, fedCh);

            conditions_->detID_[stripgpu::fedIndex(fedID)][fedCh] = detID;
            conditions_->iPair_[stripgpu::fedIndex(fedID)][fedCh] = iPair;
            conditions_->setInvThickness(fedID, fedCh, siStripClusterTools::sensorThicknessInverse(detID));

            auto offset = 256 * iPair;

            for (auto strip = 0; strip < 256; ++strip) {
              const auto gainRange = gains->getRange(det);

              const auto detstrip = strip + offset;
              const std::uint16_t noise = SiStripNoises::getRawNoise(detstrip, noises.getRange(det));
              const auto gain = SiStripGain::getStripGain(detstrip, gainRange);
              const auto bad = quality.IsStripBad(quality.getRange(det), detstrip);

              // gain is actually stored per-APV, not per-strip
              conditions_->setStrip(fedID, fedCh, strip, noise, gain, bad);
            }
          }
        }
      }
    }
  }

  std::sort(detToFeds_.begin(), detToFeds_.end(), [](const DetToFed& a, const DetToFed& b) {
    return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
  });
}

SiStripClusterizerConditionsGPU::~SiStripClusterizerConditionsGPU() {
  if (nullptr != conditions_) {
    cudaCheck(cudaFreeHost(conditions_));
  }
}

SiStripClusterizerConditionsGPU::Data const* SiStripClusterizerConditionsGPU::getGPUProductAsync(
    cudaStream_t stream) const {
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.conditionsDevice, sizeof(Data)));
    cudaCheck(cudaMemcpyAsync(data.conditionsDevice, conditions_, sizeof(Data), cudaMemcpyDefault, stream));
  });
  // Returns the payload object on the memory of the current device
  return data.conditionsDevice;
}

SiStripClusterizerConditionsGPU::GPUData::~GPUData() { cudaCheck(cudaFree(conditionsDevice)); }
