#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsGPU_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsGPU_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

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
  __host__ __device__ inline std::uint32_t stripIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return fedIndex(fed) * kStripsPerFed + channel * kStripsPerChannel + (strip % kStripsPerChannel);
  }
  __host__ __device__ inline std::uint32_t apvIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return fedIndex(fed) * kApvCount + 2 * channel + (strip % kStripsPerChannel) / 128;
  }
  __host__ __device__ inline std::uint32_t channelIndex(fedId_t fed, fedCh_t channel) {
    return fedIndex(fed) * kChannelCount + channel;
  }

  class SiStripClusterizerConditionsGPU {
  public:
    class DetToFed {
    public:
      DetToFed(detId_t detid, APVPair_t ipair, fedId_t fedid, fedCh_t fedch)
          : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
      detId_t detID() const { return detid_; }
      APVPair_t pair() const { return ipair_; }
      fedId_t fedID() const { return fedid_; }
      fedCh_t fedCh() const { return fedch_; }

    private:
      detId_t detid_;
      APVPair_t ipair_;
      fedId_t fedid_;
      fedCh_t fedch_;
    };
    using DetToFeds = std::vector<DetToFed>;

    static constexpr std::uint16_t badBit = 1 << 15;

    class Data {
    public:
      struct DeviceView {
        __device__ inline detId_t detID(fedId_t fed, fedCh_t channel) const {
          return detID_[channelIndex(fed, channel)];
        }

        __device__ inline APVPair_t iPair(fedId_t fed, fedCh_t channel) const {
          return iPair_[channelIndex(fed, channel)];
        }

        __device__ inline float invthick(fedId_t fed, fedCh_t channel) const {
          return invthick_[channelIndex(fed, channel)];
        }

        __device__ inline float noise(fedId_t fed, fedCh_t channel, stripId_t strip) const {
          return 0.1f * (noise_[stripIndex(fed, channel, strip)] & ~badBit);
        }

        __device__ inline float gain(fedId_t fed, fedCh_t channel, stripId_t strip) const {
          return gain_[apvIndex(fed, channel, strip)];
        }

        __device__ inline bool bad(fedId_t fed, fedCh_t channel, stripId_t strip) const {
          return badBit == (noise_[stripIndex(fed, channel, strip)] & badBit);
        }
        const std::uint16_t* noise_;  //[kFedCount*kStripsPerFed];
        const float* invthick_;       //[kFedCount*kChannelCount];
        const detId_t* detID_;        //[kFedCount*kChannelCount];
        const APVPair_t* iPair_;      //[kFedCount*kChannelCount];
        const float* gain_;           //[kFedCount*kApvCount];
      };

      const DeviceView* deviceView() const { return deviceView_.get(); }

      cms::cuda::device::unique_ptr<DeviceView> deviceView_;
      cms::cuda::host::unique_ptr<DeviceView> hostView_;

      cms::cuda::device::unique_ptr<std::uint16_t[]> noise_;  //[kFedCount*kStripsPerFed];
      cms::cuda::device::unique_ptr<float[]> invthick_;       //[kFedCount*kChannelCount];
      cms::cuda::device::unique_ptr<detId_t[]> detID_;        //[kFedCount*kChannelCount];
      cms::cuda::device::unique_ptr<APVPair_t[]> iPair_;      //[kFedCount*kChannelCount];
      cms::cuda::device::unique_ptr<float[]> gain_;           //[kFedCount*kApvCount];
    };

    SiStripClusterizerConditionsGPU(const SiStripQuality& quality,
                                    const SiStripGain* gains,
                                    const SiStripNoises& noises);
    ~SiStripClusterizerConditionsGPU() = default;

    // Function to return the actual payload on the memory of the current device
    Data const& getGPUProductAsync(cudaStream_t stream) const;

    const DetToFeds& detToFeds() const { return detToFeds_; }

  private:
    void setStrip(fedId_t fed, fedCh_t channel, stripId_t strip, std::uint16_t noise, float gain, bool bad) {
      gain_[apvIndex(fed, channel, strip)] = gain;
      noise_[stripIndex(fed, channel, strip)] = noise;
      if (bad) {
        noise_[stripIndex(fed, channel, strip)] |= badBit;
      }
    }

    void setInvThickness(fedId_t fed, fedCh_t channel, float invthick) {
      invthick_[channelIndex(fed, channel)] = invthick;
    }

    // Holds the data in pinned CPU memory
    std::vector<std::uint16_t, cms::cuda::HostAllocator<std::uint16_t>> noise_;
    std::vector<float, cms::cuda::HostAllocator<float>> invthick_;
    std::vector<detId_t, cms::cuda::HostAllocator<detId_t>> detID_;
    std::vector<APVPair_t, cms::cuda::HostAllocator<APVPair_t>> iPair_;
    std::vector<float, cms::cuda::HostAllocator<float>> gain_;

    // Helper that takes care of complexity of transferring the data to
    // multiple devices
    cms::cuda::ESProduct<Data> gpuData_;
    DetToFeds detToFeds_;
  };
}  // namespace stripgpu

#endif
