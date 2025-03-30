#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h

#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripMappingDevice.h"

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
    DataFedAppender(Queue& queue, unsigned int bytesN)
        : bufferSize_byte_(bytesN),
          fedIDset_(sistrip::NUMBER_OF_FEDS, false),
          pinnedBuffer_(cms::alpakatools::make_host_buffer<uint8_t[]>(queue, bytesN)),
          offsetInBuffer_(0) {};

    void insertFEDRawDataObj(uint16_t fedID, const FEDRawData* rawFEDData) {
      // Check first if the rawFEDData is empty
      if (rawFEDData->size() > 0) {
        const auto fedi = fedID - sistrip::FED_ID_MIN;
        if (fedIDset_[fedi] == true) [[unlikely]] {
          // FEDID already set
          throw cms::Exception("RawToDigi") << "FEDID " << fedID << " already previously set!";
        } else {
          // Make sure this chunk of data won't overflow the buffer
          if ((offsetInBuffer_ + rawFEDData->size()) > bufferSize_byte_) [[unlikely]] {
            throw cms::Exception("RawToDigi") << "Buffer overflow! FEDID " << fedID << " exceeds the buffer size!";
          }

          // Dereference and copy this chunk of memory into the pinned buffer
          std::memcpy(pinnedBuffer_.data() + offsetInBuffer_, rawFEDData->data(), rawFEDData->size());
          // Flag this FEDID as existing in the buffer
          fedIDset_[fedi] = true;
          // Set the current position of offsetInBuffer_ for this FEDID (direct FEDID indexing)
          fedOffsetsMap_[fedID] = offsetInBuffer_;
          // Add this chunk start index to the vector
          chunkStartIdx_.emplace_back(offsetInBuffer_);
          // Update the position within the buffer
          offsetInBuffer_ += rawFEDData->size();
        }
      }
    }

    // Is the fedID in the set?
    bool isInside(uint16_t fedID) const {
      const unsigned long fedi = fedID - sistrip::FED_ID_MIN;
      if (fedi < fedIDset_.size())
        return fedIDset_[fedi];
      return false;
    }

    // Get the offset in buffer corresponding to a given FEDID,
    // if not exists, then nullptr from the sentinel value of optional.
    inline std::optional<size_t> getOffset4FEDID(uint16_t fedID) {
      auto it = fedOffsetsMap_.find(fedID);
      if (it == fedOffsetsMap_.end())
        return std::nullopt;
      return it->second;
    }

    // Get owness of the buffer, so it can be moved
    auto getBuffer() { return std::move(pinnedBuffer_); }

    // Get the size of the preallocated buffer
    inline auto getPreallocSize() { return bufferSize_byte_; }

    // Get the current position of the offset
    inline auto getCurrentOffset() { return offsetInBuffer_; }

  private:
    const size_t bufferSize_byte_;
    std::vector<bool> fedIDset_;
    cms::alpakatools::host_buffer<uint8_t[]> pinnedBuffer_;
    size_t offsetInBuffer_;

    std::map<uint16_t, size_t> fedOffsetsMap_;
    std::vector<unsigned int> chunkStartIdx_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  class SiStripRawToClusterAlgo {
  public:
    SiStripRawToClusterAlgo(const edm::ParameterSet& unpackPar, const edm::ParameterSet& clustPar);

    void initialize(Queue& queue, int n_strips);

    void unpackStrips(Queue& queue,
                      SiStripMappingDevice const& mapping,
                      SiStripClusterizerConditionsDevice const& conditions);

    void setSeedsAndMakeIndexes(Queue& queue,
                                SiStripMappingDevice const& mapping,
                                SiStripClusterizerConditionsDevice const& conditions);
    std::unique_ptr<SiStripClustersDevice> makeClusters(Queue& queue,
                                                        SiStripMappingDevice const& mapping,
                                                        SiStripClusterizerConditionsDevice const& conditions);

  private:
    // Unpacker parameters
    const bool isLegacyUnpacker_;
    const FEDLegacyReadoutMode legacyUnpackerROmode_ = READOUT_MODE_LEGACY_INVALID;

    // Clusterizer parameters
    const float channelThreshold_, seedThreshold_, clusterThresholdSquared_;
    const uint8_t maxSequentialHoles_, maxSequentialBad_, maxAdjacentBad_;
    const uint32_t maxClusterSize_;
    const float minGoodCharge_;

    int nStripsBytes_ = 0;
    std::unique_ptr<StripDigiDevice> digis_d_;
    std::unique_ptr<StripClustersAuxDevice> sClustersAux_d_;

    void dumpUnpackedStrips(Queue& queue, StripDigiDevice* digis_d);
    void dumpSeeds(Queue& queue, StripDigiDevice* digis_d, StripClustersAuxDevice* sClustersAux_d);
    void dumpClusters(Queue& queue, SiStripClustersDevice* clusters_d);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
