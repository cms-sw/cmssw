#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

#include "SiStripRawToClusterGPUKernel.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"
#include "ChannelLocsGPU.h"
#include "StripDataView.h"

namespace stripgpu {
  StripDataGPU::StripDataGPU(size_t size, cudaStream_t stream) {
    alldataGPU_ = cms::cuda::make_device_unique<uint8_t[]>(size, stream);
    channelGPU_ = cms::cuda::make_device_unique<uint16_t[]>(size, stream);
    stripIdGPU_ = cms::cuda::make_device_unique<stripgpu::stripId_t[]>(size, stream);
  }

  SiStripRawToClusterGPUKernel::SiStripRawToClusterGPUKernel(const edm::ParameterSet& conf)
      : fedIndex_(sistrip::NUMBER_OF_FEDS, stripgpu::invalidFed),
        channelThreshold_(conf.getParameter<double>("ChannelThreshold")),
        seedThreshold_(conf.getParameter<double>("SeedThreshold")),
        clusterThresholdSquared_(std::pow(conf.getParameter<double>("ClusterThreshold"), 2.0f)),
        maxSequentialHoles_(conf.getParameter<unsigned>("MaxSequentialHoles")),
        maxSequentialBad_(conf.getParameter<unsigned>("MaxSequentialBad")),
        maxAdjacentBad_(conf.getParameter<unsigned>("MaxAdjacentBad")),
        maxClusterSize_(conf.getParameter<unsigned>("MaxClusterSize")),
        minGoodCharge_(clusterChargeCut(conf)) {
    fedRawDataOffsets_.reserve(sistrip::NUMBER_OF_FEDS);
  }

  void SiStripRawToClusterGPUKernel::makeAsync(const std::vector<const FEDRawData*>& rawdata,
                                               const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                                               const SiStripClusterizerConditionsGPU& conditions,
                                               cudaStream_t stream) {
    size_t totalSize{0};
    for (const auto& buff : buffers) {
      if (buff != nullptr) {
        totalSize += buff->bufferSize();
      }
    }

    auto fedRawDataHost = cms::cuda::make_host_unique<uint8_t[]>(totalSize, stream);
    auto fedRawDataGPU = cms::cuda::make_device_unique<uint8_t[]>(totalSize, stream);

    size_t off = 0;
    fedRawDataOffsets_.clear();
    fedIndex_.clear();
    fedIndex_.resize(sistrip::NUMBER_OF_FEDS, stripgpu::invalidFed);

    sistrip::FEDReadoutMode mode = sistrip::READOUT_MODE_INVALID;

    for (size_t fedi = 0; fedi < buffers.size(); ++fedi) {
      auto& buff = buffers[fedi];
      if (buff != nullptr) {
        const auto raw = rawdata[fedi];
        memcpy(fedRawDataHost.get() + off, raw->data(), raw->size());
        fedIndex_[stripgpu::fedIndex(fedi)] = fedRawDataOffsets_.size();
        fedRawDataOffsets_.push_back(off);
        off += raw->size();
        if (fedRawDataOffsets_.size() == 1) {
          mode = buff->readoutMode();
        } else {
          if (buff->readoutMode() != mode) {
            throw cms::Exception("[SiStripRawToClusterGPUKernel] inconsistent readout mode ")
                << buff->readoutMode() << " != " << mode;
          }
        }
      }
    }
    // send rawdata to GPU
    cms::cuda::copyAsync(fedRawDataGPU, fedRawDataHost, totalSize, stream);

    const auto& detmap = conditions.detToFeds();
    if ((mode != sistrip::READOUT_MODE_ZERO_SUPPRESSED) && (mode != sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE10)) {
      throw cms::Exception("[SiStripRawToClusterGPUKernel] unsupported readout mode ") << mode;
    }
    const uint16_t headerlen = mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;
    size_t offset = 0;
    auto chanlocs = std::make_unique<ChannelLocs>(detmap.size(), stream);
    auto inputGPU = cms::cuda::make_host_unique<const uint8_t*[]>(chanlocs->size(), stream);

    // iterate over the detector in DetID/APVPair order
    // mapping out where the data are
    for (size_t i = 0; i < detmap.size(); ++i) {
      const auto& detp = detmap[i];
      const auto fedId = detp.fedID();
      const auto fedCh = detp.fedCh();
      const auto fedi = fedIndex_[stripgpu::fedIndex(fedId)];

      if (fedi != invalidFed) {
        const auto buffer = buffers[fedId].get();
        const auto& channel = buffer->channel(detp.fedCh());

        auto len = channel.length();
        auto off = channel.offset();

        assert(len >= headerlen || len == 0);

        if (len >= headerlen) {
          len -= headerlen;
          off += headerlen;
        }

        chanlocs->setChannelLoc(i, channel.data(), off, offset, len, fedId, fedCh, detp.detID());
        inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets_[fedi] + (channel.data() - rawdata[fedId]->data());
        offset += len;

      } else {
        chanlocs->setChannelLoc(i, nullptr, 0, 0, 0, invalidFed, 0, invalidDet);
        inputGPU[i] = nullptr;
      }
    }

    const auto n_strips = offset;

    sst_data_d_ = cms::cuda::make_host_unique<StripDataView>(stream);
    sst_data_d_->nStrips = n_strips;

    chanlocsGPU_ = std::make_unique<ChannelLocsGPU>(detmap.size(), stream);
    chanlocsGPU_->setVals(chanlocs.get(), std::move(inputGPU), stream);

    stripdata_ = std::make_unique<StripDataGPU>(n_strips, stream);

    const auto& condGPU = conditions.getGPUProductAsync(stream);

    unpackChannelsGPU(condGPU.deviceView(), stream);
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif

#ifdef EDM_ML_DEBUG
    auto outdata = cms::cuda::make_host_unique<uint8_t[]>(n_strips, stream);
    cms::cuda::copyAsync(outdata, stripdata_->alldataGPU_, n_strips, stream);
    cudaCheck(cudaStreamSynchronize(stream));

    constexpr int xor3bits = 7;
    for (size_t i = 0; i < chanlocs->size(); ++i) {
      const auto data = chanlocs->input(i);
      const auto len = chanlocs->length(i);

      if (data != nullptr && len > 0) {
        auto aoff = chanlocs->offset(i);
        auto choff = chanlocs->inoff(i);
        const auto end = choff + len;

        while (choff < end) {
          const auto stripIndex = data[choff++ ^ xor3bits];
          const auto groupLength = data[choff++ ^ xor3bits];
          aoff += 2;
          for (auto k = 0; k < groupLength; ++k, ++choff, ++aoff) {
            if (data[choff ^ xor3bits] != outdata[aoff]) {
              LogDebug("SiStripRawToClusterGPUKernel")
                  << "Strip mismatch " << stripIndex << " i:k " << i << ":" << k << " "
                  << (uint32_t)data[choff ^ xor3bits] << " != " << (uint32_t)outdata[aoff] << std::endl;
            }
          }
        }
      }
    }
    outdata.reset(nullptr);
#endif

    fedRawDataGPU.reset();
    allocateSSTDataGPU(n_strips, stream);
    setSeedStripsNCIndexGPU(condGPU.deviceView(), stream);

    clusters_d_ = SiStripClustersCUDADevice(kMaxSeedStrips, maxClusterSize_, stream);
    findClusterGPU(condGPU.deviceView(), stream);

    stripdata_.reset();
  }

  SiStripClustersCUDADevice SiStripRawToClusterGPUKernel::getResults(cudaStream_t stream) {
    reset();

    return std::move(clusters_d_);
  }

  void SiStripRawToClusterGPUKernel::reset() {
    chanlocsGPU_.reset();
    sst_data_d_.reset();
  }
}  // namespace stripgpu
