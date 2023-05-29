#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

namespace stripgpu {
  SiStripClusterizerConditionsGPU::SiStripClusterizerConditionsGPU(const SiStripQuality& quality,
                                                                   const SiStripGain* gains,
                                                                   const SiStripNoises& noises)

      : noise_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH),
        invthick_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED),
        detID_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED),
        iPair_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED),
        gain_(sistrip::NUMBER_OF_FEDS * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED) {
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

              detID_[channelIndex(fedID, fedCh)] = detID;
              iPair_[channelIndex(fedID, fedCh)] = iPair;
              setInvThickness(fedID, fedCh, siStripClusterTools::sensorThicknessInverse(detID));

              auto offset = 256 * iPair;

              for (auto strip = 0; strip < 256; ++strip) {
                const auto gainRange = gains->getRange(det);

                const auto detstrip = strip + offset;
                const std::uint16_t noise = SiStripNoises::getRawNoise(detstrip, noises.getRange(det));
                const auto gain = SiStripGain::getStripGain(detstrip, gainRange);
                const auto bad = quality.IsStripBad(quality.getRange(det), detstrip);

                // gain is actually stored per-APV, not per-strip
                setStrip(fedID, fedCh, detstrip, noise, gain, bad);
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

  SiStripClusterizerConditionsGPU::Data const& SiStripClusterizerConditionsGPU::getGPUProductAsync(
      cudaStream_t stream) const {
    auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](Data& data, cudaStream_t stream) {
      data.noise_ = cms::cuda::make_device_unique<std::uint16_t[]>(noise_.size(), stream);
      data.invthick_ = cms::cuda::make_device_unique<float[]>(invthick_.size(), stream);
      data.detID_ = cms::cuda::make_device_unique<detId_t[]>(detID_.size(), stream);
      data.iPair_ = cms::cuda::make_device_unique<apvPair_t[]>(iPair_.size(), stream);
      data.gain_ = cms::cuda::make_device_unique<float[]>(gain_.size(), stream);

      cms::cuda::copyAsync(data.noise_, noise_, stream);
      cms::cuda::copyAsync(data.invthick_, invthick_, stream);
      cms::cuda::copyAsync(data.detID_, detID_, stream);
      cms::cuda::copyAsync(data.iPair_, iPair_, stream);
      cms::cuda::copyAsync(data.gain_, gain_, stream);

      data.hostView_ = cms::cuda::make_host_unique<SiStripClusterizerConditionsGPU::Data::DeviceView>(stream);
      data.hostView_->noise_ = data.noise_.get();
      data.hostView_->invthick_ = data.invthick_.get();
      data.hostView_->detID_ = data.detID_.get();
      data.hostView_->iPair_ = data.iPair_.get();
      data.hostView_->gain_ = data.gain_.get();

      data.deviceView_ = cms::cuda::make_device_unique<SiStripClusterizerConditionsGPU::Data::DeviceView>(stream);
      cms::cuda::copyAsync(data.deviceView_, data.hostView_, stream);
    });

    return data;
  }
}  // namespace stripgpu
