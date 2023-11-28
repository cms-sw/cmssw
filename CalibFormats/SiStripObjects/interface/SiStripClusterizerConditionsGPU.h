#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsGPU_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsGPU_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

class SiStripQuality;
class SiStripGain;
class SiStripNoises;

namespace stripgpu {
  __host__ __device__ inline fedId_t fedIndex(fedId_t fed) { return fed - sistrip::FED_ID_MIN; }
  __host__ __device__ inline std::uint32_t stripIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return fedIndex(fed) * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH + channel * sistrip::STRIPS_PER_FEDCH +
           (strip % sistrip::STRIPS_PER_FEDCH);
  }
  __host__ __device__ inline std::uint32_t apvIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return fedIndex(fed) * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED + sistrip::APVS_PER_CHAN * channel +
           (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV;
  }
  __host__ __device__ inline std::uint32_t channelIndex(fedId_t fed, fedCh_t channel) {
    return fedIndex(fed) * sistrip::FEDCH_PER_FED + channel;
  }

  class SiStripClusterizerConditionsGPU {
  public:
    class DetToFed {
    public:
      DetToFed(detId_t detid, apvPair_t ipair, fedId_t fedid, fedCh_t fedch)
          : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
      detId_t detID() const { return detid_; }
      apvPair_t pair() const { return ipair_; }
      fedId_t fedID() const { return fedid_; }
      fedCh_t fedCh() const { return fedch_; }

    private:
      detId_t detid_;
      apvPair_t ipair_;
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

        __device__ inline apvPair_t iPair(fedId_t fed, fedCh_t channel) const {
          return iPair_[channelIndex(fed, channel)];
        }

        __device__ inline float invthick(fedId_t fed, fedCh_t channel) const {
          return invthick_[channelIndex(fed, channel)];
        }

        __device__ inline float noise(fedId_t fed, fedCh_t channel, stripId_t strip) const {
          // noise is stored as 9 bits with a fixed point scale factor of 0.1
          return 0.1f * (noise_[stripIndex(fed, channel, strip)] & ~badBit);
        }

        __device__ inline float gain(fedId_t fed, fedCh_t channel, stripId_t strip) const {
          return gain_[apvIndex(fed, channel, strip)];
        }

        __device__ inline bool bad(fedId_t fed, fedCh_t channel, stripId_t strip) const {
          return badBit == (noise_[stripIndex(fed, channel, strip)] & badBit);
        }
        const std::uint16_t* noise_;  //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH];
        const float* invthick_;       //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
        const detId_t* detID_;        //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
        const apvPair_t* iPair_;      //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
        const float* gain_;           //[sistrip::NUMBER_OF_FEDS*sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED];
      };

      const DeviceView* deviceView() const { return deviceView_.get(); }

      cms::cuda::device::unique_ptr<DeviceView> deviceView_;
      cms::cuda::host::unique_ptr<DeviceView> hostView_;

      cms::cuda::device::unique_ptr<std::uint16_t[]>
          noise_;  //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH];
      cms::cuda::device::unique_ptr<float[]> invthick_;   //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
      cms::cuda::device::unique_ptr<detId_t[]> detID_;    //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
      cms::cuda::device::unique_ptr<apvPair_t[]> iPair_;  //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
      cms::cuda::device::unique_ptr<float[]>
          gain_;  //[sistrip::NUMBER_OF_FEDS*sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED];
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
    std::vector<apvPair_t, cms::cuda::HostAllocator<apvPair_t>> iPair_;
    std::vector<float, cms::cuda::HostAllocator<float>> gain_;

    // Helper that takes care of complexity of transferring the data to
    // multiple devices
    cms::cuda::ESProduct<Data> gpuData_;
    DetToFeds detToFeds_;
  };
}  // namespace stripgpu

#endif
