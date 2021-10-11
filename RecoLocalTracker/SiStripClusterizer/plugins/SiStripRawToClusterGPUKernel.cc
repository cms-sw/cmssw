#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

#include "SiStripRawToClusterGPUKernel.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"
#include "ChannelLocsGPU.h"
#include "StripDataView.cuh"

namespace stripgpu {
  StripDataGPU::StripDataGPU(size_t size, cudaStream_t stream) {
    alldataGPU_ = cms::cuda::make_device_unique<uint8_t[]>(size, stream);
    channelGPU_ = cms::cuda::make_device_unique<uint16_t[]>(size, stream);
    stripIdGPU_ = cms::cuda::make_device_unique<stripgpu::stripId_t[]>(size, stream);
  }

  SiStripRawToClusterGPUKernel::SiStripRawToClusterGPUKernel(const edm::ParameterSet& conf)
      : fedIndex_(stripgpu::kFedCount, stripgpu::invalidFed),
        channelThreshold_(conf.getParameter<double>("ChannelThreshold")),
        seedThreshold_(conf.getParameter<double>("SeedThreshold")),
        clusterThresholdSquared_(std::pow(conf.getParameter<double>("ClusterThreshold"), 2.0f)),
        maxSequentialHoles_(conf.getParameter<unsigned>("MaxSequentialHoles")),
        maxSequentialBad_(conf.getParameter<unsigned>("MaxSequentialBad")),
        maxAdjacentBad_(conf.getParameter<unsigned>("MaxAdjacentBad")),
        maxClusterSize_(conf.getParameter<unsigned>("MaxClusterSize")),
        minGoodCharge_(clusterChargeCut(conf)) {
    fedRawDataOffsets_.reserve(stripgpu::kFedCount);
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

    fedRawDataHost_ = cms::cuda::make_host_unique<uint8_t[]>(totalSize, stream);
    auto fedRawDataGPU = cms::cuda::make_device_unique<uint8_t[]>(totalSize, stream);

    size_t off = 0;
    fedRawDataOffsets_.clear();
    fedIndex_.clear();
    fedIndex_.resize(stripgpu::kFedCount, stripgpu::invalidFed);

    sistrip::FEDReadoutMode mode = sistrip::READOUT_MODE_INVALID;

    for (size_t fedi = 0; fedi < buffers.size(); ++fedi) {
      auto& buff = buffers[fedi];
      if (buff != nullptr) {
        const auto raw = rawdata[fedi];
        memcpy(fedRawDataHost_.get() + off, raw->data(), raw->size());
        fedIndex_[stripgpu::fedIndex(fedi)] = fedRawDataOffsets_.size();
        fedRawDataOffsets_.push_back(off);
        off += raw->size();
        if (fedRawDataOffsets_.size() == 1) {
          mode = buff->readoutMode();
        } else {
          assert(buff->readoutMode() == mode);
        }
      }
    }
    // send rawdata to GPU
    cms::cuda::copyAsync(fedRawDataGPU, fedRawDataHost_, totalSize, stream);

    const auto& detmap = conditions.detToFeds();
    const uint16_t headerlen = mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;
    size_t offset = 0;
    chanlocs_ = std::make_unique<ChannelLocs>(detmap.size(), stream);
    std::vector<uint8_t*> inputGPU(chanlocs_->size());

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

        chanlocs_->setChannelLoc(i, channel.data(), off, offset, len, fedId, fedCh, detp.detID());
        inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets_[fedi] + (channel.data() - rawdata[fedId]->data());
        offset += len;

      } else {
        chanlocs_->setChannelLoc(i, nullptr, 0, 0, 0, invalidFed, 0, invalidDet);
        inputGPU[i] = nullptr;
      }
    }

    const auto max_strips = offset;

    sst_data_d_ = cms::cuda::make_host_unique<StripDataView>(stream);
    sst_data_d_->nStrips = max_strips;

    chanlocsGPU_ = std::make_unique<ChannelLocsGPU>(detmap.size(), stream);
    chanlocsGPU_->setVals(chanlocs_.get(), inputGPU, stream);

    stripdata_ = std::make_unique<StripDataGPU>(max_strips, stream);
    const int max_seedstrips = kMaxSeedStrips;

    const auto& condGPU = conditions.getGPUProductAsync(stream);

    unpackChannelsGPU(condGPU.deviceView(), stream);

#ifdef EDM_ML_DEBUG
    auto outdata = cms::cuda::make_host_unique<uint8_t[]>(max_strips, stream);
    cms::cuda::copyAsync(outdata, stripdata_->alldataGPU_, max_strips, stream);
    cudaCheck(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < chanlocs_->size(); ++i) {
      const auto data = chanlocs_->input(i);
      const auto len = chanlocs_->length(i);

      if (data != nullptr && len > 0) {
        auto aoff = chanlocs_->offset(i);
        auto choff = chanlocs_->inoff(i);
        const auto end = choff + len;

        while (choff < end) {
          const auto stripIndex = data[choff++ ^ 7];
          const auto groupLength = data[choff++ ^ 7];
          aoff += 2;
          for (auto k = 0; k < groupLength; ++k, ++choff, ++aoff) {
            if (data[choff ^ 7] != outdata[aoff]) {
              LogDebug("SiStripRawToClusterGPUKernel")
                  << "Strip mismatch " << stripIndex << " i:k " << i << ":" << k << " " << (uint32_t)data[choff ^ 7]
                  << " != " << (uint32_t)outdata[aoff] << std::endl;
            }
          }
        }
      }
    }
    outdata.reset(nullptr);
#endif

    fedRawDataGPU.reset();
    allocateSSTDataGPU(max_strips, stream);
    setSeedStripsNCIndexGPU(condGPU.deviceView(), stream);

    clusters_d_ = SiStripClustersCUDADevice(max_seedstrips, maxClusterSize_, stream);
    findClusterGPU(condGPU.deviceView(), stream);

    stripdata_.reset();
    chanlocsGPU_.reset();
  }

  SiStripClustersCUDADevice SiStripRawToClusterGPUKernel::getResults(cudaStream_t stream) {
    reset();

    return std::move(clusters_d_);
  }

  void SiStripRawToClusterGPUKernel::reset() {
    fedRawDataHost_.reset();
    chanlocs_.reset();
    sst_data_d_.reset();
  }
}  // namespace stripgpu
