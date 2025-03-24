#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h

#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripMappingDevice.h"

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace edm {
  class ParameterSet;
}

namespace sistripclusterizer {
  GENERATE_SOA_LAYOUT(StripDigiSoALayout,
                      SOA_COLUMN(uint8_t, adc),
                      SOA_COLUMN(uint16_t, channel),
                      SOA_COLUMN(uint16_t, stripId))

  GENERATE_SOA_LAYOUT(StripClustersAuxSoALayout,
                      SOA_COLUMN(int, seedStripsMask),
                      SOA_COLUMN(int, seedStripsNCMask),
                      SOA_COLUMN(int, seedStripsNCIndex),
                      SOA_COLUMN(int, prefixSeedStripsNCMask),
                      //
                      SOA_SCALAR(float, channelThreshold),
                      SOA_SCALAR(float, seedThreshold),
                      SOA_SCALAR(float, clusterThresholdSquared),
                      SOA_SCALAR(uint8_t, maxSequentialHoles),
                      SOA_SCALAR(uint8_t, maxSequentialBad),
                      SOA_SCALAR(uint8_t, maxAdjacentBad),
                      // SOA_SCALAR(uint32_t, maxClusterSize),
                      SOA_SCALAR(float, minGoodCharge),
                      SOA_SCALAR(int, clusterSizeLimit))

  using StripDigiSoA = StripDigiSoALayout<>;
  using StripDigiView = StripDigiSoA::View;
  using StripDigiConstView = StripDigiSoA::ConstView;
  using StripDigiHost = PortableHostCollection<StripDigiSoA>;

  using StripClustersAuxSoA = StripClustersAuxSoALayout<>;
  using StripClustersAuxView = StripClustersAuxSoA::View;
  using StripClustersAuxConstView = StripClustersAuxSoA::ConstView;
  using StripClustersAuxHost = PortableHostCollection<StripClustersAuxSoA>;

  using StripClusterizerHost = PortableHostCollection2<StripDigiSoA, StripClustersAuxSoA>;
}  // namespace sistripclusterizer

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace sistripclusterizer;

  // PortableCollection-based model
  using StripDigiHost = ::sistripclusterizer::StripDigiHost;
  using StripDigiDevice = PortableCollection<StripDigiSoA>;

  using StripClustersAuxHost = ::sistripclusterizer::StripClustersAuxHost;
  using StripClustersAuxDevice = PortableCollection<StripClustersAuxSoA>;

  using StripClusterizerHost = ::sistripclusterizer::StripClusterizerHost;
  using StripClusterizerDevice = PortableCollection2<StripDigiSoA, StripClustersAuxSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

// check that the sistrip device collection for the host device is the same as the sistrip host collection
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::StripDigiDevice, sistripclusterizer::StripDigiHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::StripClustersAuxDevice, sistripclusterizer::StripClustersAuxHost);
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  /**
  * @brief Calculates the next integer.
  *
  * Helper class to store the FEW raw data which will be copyed to the device for unpacking. It contains 
  *
  * @param queue The Queue object to book the host buffers (typically the iEvent.queue())
  * @param bufferSize_bytes The extent of the buffer size in bytes
  */
  class DataFedAppender {
  public:
    DataFedAppender(Queue& queue, unsigned int bufferSize_bytes)
        : bytes_{cms::alpakatools::make_host_buffer<uint8_t[]>(queue, bufferSize_bytes)},
          // fedID_{cms::alpakatools::make_host_buffer<uint16_t[]>(queue, bufferSize_bytes)},
          fedIDinSet_(sistrip::NUMBER_OF_FEDS, false),
          size_(bufferSize_bytes / sizeof(uint8_t)),
          offset_(0) {};

    void insertFEDRawDataObj(uint16_t fedID, const FEDRawData* rawFEDData) {
      // assert((offset_+rawFEDData->size()) < size_);
      std::memcpy(bytes_.data() + offset_, rawFEDData->data(), rawFEDData->size());
      // std::memset(fedID_.data() + offset_, fedID, rawFEDData->size() * sizeof(uint16_t));

      validFED_offsets[fedID] = offset_;
      chunkStartIdx_.emplace_back(offset_);

      offset_ += rawFEDData->size();
      fedIDinSet_[fedID - sistrip::FED_ID_MIN] = true;
    }

    auto getData() const { return bytes_; }
    inline auto getSize() const { return size_; }
    inline auto getOffset() const { return offset_; }
    // inline auto getFEDIdMask() const { return fedID_; }
    inline auto chunkStartIdx() const { return chunkStartIdx_; }

    bool isInside(uint16_t fedID) const {
      uint16_t fedi = fedID - sistrip::FED_ID_MIN;
      if (fedi >= fedIDinSet_.size()) {
        return false;
      } else {
        return fedIDinSet_[fedi];
      }
    }

    inline size_t getOffset(uint16_t fedID) { return validFED_offsets[fedID]; }

    inline auto size() { return size_; }

#ifdef EDM_ML_DEBUG
    void print_Info() {
      LogDebug("DataFedAppender")
          << "There are " << validFED_offsets.size()
          << " valid FED channel buffers. The list of FEDID and offsets in the memory buffer are:\nfedID\toffset\n";
#ifdef SUPERDETAILS
      for (const auto& item : validFED_offsets) {
        LogDebug("DataFedAppender") << item.first << "\t" << item.second << "\n";
      }
#endif
    }
#endif

  private:
    cms::alpakatools::host_buffer<uint8_t[]> bytes_;
    // cms::alpakatools::host_buffer<uint16_t[]> fedID_;
    std::vector<bool> fedIDinSet_;
    const size_t size_;
    size_t offset_;

    std::map<uint16_t, size_t> validFED_offsets;
    std::vector<unsigned int> chunkStartIdx_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  class SiStripRawToClusterAlgo {
  public:
    SiStripRawToClusterAlgo(const edm::ParameterSet& conf, bool legacyMode);

    void initialize(Queue& queue, int n_strips);
    void unpackStrips(Queue& queue,
                      SiStripMappingDevice const& mapping,
                      SiStripClusterizerConditionsDevice const& conditions);
    void setSeedsAndMakeIndexes(Queue& queue,
                                SiStripMappingDevice const& mapping,
                                SiStripClusterizerConditionsDevice const& conditions);
    void makeClusters(Queue& queue,
                      SiStripMappingDevice const& mapping,
                      SiStripClusterizerConditionsDevice const& conditions);

    inline auto getClustersDevice() { return std::move(clusters_d_.value()); };

  private:
    // inputs for the unpacking and clustering
    std::optional<SiStripMappingDevice> chanlocs_d_;
    std::optional<SiStripClusterizerConditionsDevice> clusterizerConditions_d_;

    // portablecollection2 with auxiliary data used by the clusterizer
    std::optional<StripClusterizerDevice> clustersAux_d_;
    // portable collection for the clusters
    std::optional<sistrip::SiStripClustersDevice> clusters_d_;

    const float channelThreshold_, seedThreshold_, clusterThresholdSquared_;
    const uint8_t maxSequentialHoles_, maxSequentialBad_, maxAdjacentBad_;
    const uint32_t maxClusterSize_;
    const float minGoodCharge_;

    //
    const bool legacyUnpacker_;
    const FEDLegacyReadoutMode lmode_ = READOUT_MODE_LEGACY_INVALID;

  #ifdef EDM_ML_DEBUG
    void checkUnpackedStrips_(Queue& queue, StripClusterizerDevice& output) const;
    void checkPrefixSum_(Queue& queue, StripClusterizerDevice& output) const;
    void checkClusters_(Queue& queue) const;
  #endif
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
