#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClusterDevice.h"
#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDeviceObject.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripMappingDevice.h"

namespace edm {
  class ParameterSet;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The StripClustersAuxSoALayout is an auxiliary SoA, with the same size of the SiStripDigiSoA, which helps
// in the clusterization process.
namespace sistrip {
  GENERATE_SOA_LAYOUT(StripClustersAuxSoALayout,
                      SOA_COLUMN(uint32_t, seedStripsMask),
                      SOA_COLUMN(uint32_t, seedStripsNCMask),
                      SOA_COLUMN(uint32_t, seedStripsNCIndex),
                      SOA_COLUMN(uint32_t, prefixSeedStripsNCMask),
                      //
                      SOA_SCALAR(float, channelThreshold),
                      SOA_SCALAR(float, seedThreshold),
                      SOA_SCALAR(float, clusterThresholdSquared),
                      SOA_SCALAR(uint8_t, maxSequentialHoles),
                      SOA_SCALAR(uint8_t, maxSequentialBad),
                      SOA_SCALAR(uint8_t, maxAdjacentBad),
                      SOA_SCALAR(float, minGoodCharge),
                      SOA_SCALAR(uint16_t, clusterSizeLimit))

  using StripClustersAuxSoA = StripClustersAuxSoALayout<>;
  using StripClustersAuxView = StripClustersAuxSoA::View;
  using StripClustersAuxConstView = StripClustersAuxSoA::ConstView;
  using StripClustersAuxHost = PortableHostCollection<StripClustersAuxSoA>;
}  // namespace sistrip

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  using StripClustersAuxDevice = PortableCollection<StripClustersAuxSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::StripClustersAuxDevice, sistrip::StripClustersAuxHost);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  struct FEDChMetadata {
    uint32_t detId;
    uint16_t fedId;
    uint8_t fedCh;
    //
    uint16_t fedChOfs;
    uint32_t fedChOfs_wrt_rawFedId;
    uint8_t fedBufferRO;
  };

  class PortableFEDMover {
  public:
    PortableFEDMover(Queue& queue, uint32_t rawBufferSize, uint32_t fedChannelsNb)
        : buffer_(cms::alpakatools::make_host_buffer<uint8_t[]>(queue, rawBufferSize)),
          mapping_(queue, fedChannelsNb),
          bufferSize_(0),
          channelNb_(fedChannelsNb),
          offset4FedId_(sistrip::FED_ID_MAX + 1) {}

    void fillBuffer(const std::vector<const FEDRawData*>& raw) {
      for (uint16_t fedId = sistrip::FED_ID_MIN; fedId <= sistrip::FED_ID_MAX; ++fedId) {
        if (raw[fedId]) {
          std::memcpy(buffer_.data() + bufferSize_, raw[fedId]->data(), raw[fedId]->size());
          offset4FedId_[fedId] = bufferSize_;
          bufferSize_ += raw[fedId]->size();
        }
      }
    }

    void fillMapping(const std::vector<FEDChMetadata>& channelsMeta) {
      for (uint32_t i = 0; i < channelNb_; i++) {
        const auto& channel = channelsMeta[i];
        mapping_->detID(i) = channel.detId;
        mapping_->fedID(i) = channel.fedId;
        mapping_->fedCh(i) = channel.fedCh;
        mapping_->fedChOfs(i) = channel.fedChOfs;
        mapping_->fedChDataOfsBuf(i) = channel.fedChOfs_wrt_rawFedId + offset4FedId_[channel.fedId];
        mapping_->readoutMode(i) = channel.fedBufferRO;
      }
    }

    auto buffer() { return buffer_; }
    uint32_t bufferSize() const { return bufferSize_; }

    auto mapping() { return std::move(mapping_); }
    uint32_t channelNb() const { return channelNb_; }

  private:
    cms::alpakatools::host_buffer<uint8_t[]> buffer_;
    SiStripMappingHost mapping_;

    uint32_t bufferSize_;
    uint32_t channelNb_;
    std::vector<uint32_t> offset4FedId_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  class SiStripRawToClusterAlgo {
  public:
    SiStripRawToClusterAlgo(const edm::ParameterSet& clustPar);
    void prepareUnpackCluster(Queue& queue,
                              const DetToFeds* conditions_DetToFeds,
                              std::unique_ptr<PortableFEDMover> FEDChMover);
    uint32_t unpackStrips(Queue& queue, const GainNoiseCals* calibs);

    std::unique_ptr<SiStripClusterDevice> makeClusters(Queue& queue, const GainNoiseCals* calibs);
    std::unique_ptr<SiStripDigiDevice> releaseDigiAmplitudes();

  private:
    // Clusterizer parameters
    const float channelThreshold_, seedThreshold_, clusterThresholdSquared_;
    const uint8_t maxSequentialHoles_, maxSequentialBad_, maxAdjacentBad_;
    const uint32_t maxClusterSize_;
    const float minGoodCharge_;
    const uint32_t kMaxSeedStrips_;

    std::unique_ptr<PortableFEDMover> fedChMover_;

    std::optional<cms::alpakatools::device_buffer<Device, uint8_t[]>> fedBuffer_d_;
    std::optional<SiStripMappingDevice> stripMapping_d_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> nStrips_h_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> nSeeds_h_;

    int nStripsBytes_ = 0;
    std::unique_ptr<SiStripDigiDevice> digis_d_;
    std::unique_ptr<StripClustersAuxDevice> sClustersAux_d_;

    // Debug functions
    void dumpUnpackedStrips(Queue& queue, SiStripDigiDevice* digis_d);
    void dumpSeeds(Queue& queue,
                   SiStripDigiDevice* digis_d,
                   StripClustersAuxDevice* sClustersAux_d,
                   SiStripMappingDevice* mapping_d);
    void dumpClusters(Queue& queue, SiStripClusterDevice* clusters_d, SiStripDigiDevice* digis_d, bool fullDump = false);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
