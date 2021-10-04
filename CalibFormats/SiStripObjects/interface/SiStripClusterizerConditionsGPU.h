#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsGPU_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsGPU_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"

class SiStripQuality;
class SiStripGain;
class SiStripNoises;

namespace stripgpu {
  static constexpr int kStripsPerChannel = 256;
  static constexpr int kFedFirst = 50;
  static constexpr int kFedLast = 489;
  static constexpr int kFedCount = kFedLast - kFedFirst + 1;
  static constexpr int kChannelCount = 96;
  static constexpr int kApvCount = 2 * kChannelCount;
  static constexpr int kStripsPerFed = kChannelCount * kStripsPerChannel;

  __host__ __device__ inline fedId_t fedIndex(fedId_t fed) { return fed - kFedFirst; }
  __host__ __device__ inline stripId_t stripIndex(fedCh_t channel, stripId_t strip) {
    return channel * kStripsPerChannel + (strip % kStripsPerChannel);
  }
  __host__ __device__ inline stripId_t apvIndex(fedCh_t channel, stripId_t strip) {
    return channel * kStripsPerChannel + (strip % kStripsPerChannel) / 128;
  }
}  // namespace stripgpu

class SiStripClusterizerConditionsGPU {
public:
  class DetToFed {
  public:
    DetToFed(stripgpu::detId_t detid, stripgpu::APVPair_t ipair, stripgpu::fedId_t fedid, stripgpu::fedCh_t fedch)
        : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
    stripgpu::detId_t detID() const { return detid_; }
    stripgpu::APVPair_t pair() const { return ipair_; }
    stripgpu::fedId_t fedID() const { return fedid_; }
    stripgpu::fedCh_t fedCh() const { return fedch_; }

  private:
    stripgpu::detId_t detid_;
    stripgpu::APVPair_t ipair_;
    stripgpu::fedId_t fedid_;
    stripgpu::fedCh_t fedch_;
  };
  using DetToFeds = std::vector<DetToFed>;

  struct Data {
    static constexpr std::uint16_t badBit = 1 << 15;

    __host__ __device__ void setStrip(stripgpu::fedId_t fed,
                                      stripgpu::fedCh_t channel,
                                      stripgpu::stripId_t strip,
                                      std::uint16_t noise,
                                      float gain,
                                      bool bad) {
      gain_[stripgpu::fedIndex(fed)][stripgpu::apvIndex(channel, strip)] = gain;
      noise_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] = noise;
      if (bad) {
        noise_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] |= badBit;
      }
    }

    __host__ __device__ void setInvThickness(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, float invthick) {
      invthick_[stripgpu::fedIndex(fed)][channel] = invthick;
    }

    __host__ __device__ stripgpu::detId_t detID(stripgpu::fedId_t fed, stripgpu::fedCh_t channel) const {
      return detID_[stripgpu::fedIndex(fed)][channel];
    }

    __host__ __device__ stripgpu::APVPair_t iPair(stripgpu::fedId_t fed, stripgpu::fedCh_t channel) const {
      return iPair_[stripgpu::fedIndex(fed)][channel];
    }

    __host__ __device__ float invthick(stripgpu::fedId_t fed, stripgpu::fedCh_t channel) const {
      return invthick_[stripgpu::fedIndex(fed)][channel];
    }

    __host__ __device__ float noise(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, stripgpu::stripId_t strip) const {
      return 0.1 * (noise_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] & !badBit);
    }

    __host__ __device__ float gain(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, stripgpu::stripId_t strip) const {
      return gain_[stripgpu::fedIndex(fed)][stripgpu::apvIndex(channel, strip)];
    }

    __host__ __device__ bool bad(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, stripgpu::stripId_t strip) const {
      return badBit == (noise_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] & badBit);
    }

    alignas(128) float gain_[stripgpu::kFedCount][stripgpu::kApvCount];
    alignas(128) float invthick_[stripgpu::kFedCount][stripgpu::kChannelCount];
    alignas(128) std::uint16_t noise_[stripgpu::kFedCount][stripgpu::kStripsPerFed];
    alignas(128) stripgpu::detId_t detID_[stripgpu::kFedCount][stripgpu::kChannelCount];
    alignas(128) stripgpu::APVPair_t iPair_[stripgpu::kFedCount][stripgpu::kChannelCount];
  };

  SiStripClusterizerConditionsGPU(const SiStripQuality& quality, const SiStripGain* gains, const SiStripNoises& noises);
  ~SiStripClusterizerConditionsGPU();

  // Function to return the actual payload on the memory of the current device
  Data const* getGPUProductAsync(cudaStream_t stream) const;

  const DetToFeds& detToFeds() const { return detToFeds_; }

private:
  // Holds the data in pinned CPU memory
  Data* conditions_ = nullptr;

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    Data* conditionsDevice = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
  DetToFeds detToFeds_;
};

#endif
