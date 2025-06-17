#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClusterDevice.h"
#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDevice.h"
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

  namespace fedChannelDetails {
    // It corresponds to the one in EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h,
    // but for Alpaka, with the addition of stripsInCh function
    class FEDChannel {
    public:
      ALPAKA_FN_HOST_ACC inline FEDChannel(const uint8_t* data, uint32_t offset, bool isNonLite)
          : data_(data), offset_(offset), headerLen_(isNonLite ? 7 : 2) {
        length_ = (data_[(offset_) ^ 7] + (data_[(offset_ + 1) ^ 7] << 8));
      }

      ALPAKA_FN_HOST_ACC inline uint16_t length() const { return length_; }

      ALPAKA_FN_HOST_ACC inline uint16_t stripsInCh(uint8_t num_bits = 8) const {
        const bool emptyCh = (headerLen_ + 2) >= (length_);
        const uint16_t start = offset_ + headerLen_;
        const uint16_t end = offset_ + length_;
        uint16_t stripN = 0;
        if (!emptyCh) {
          for (uint16_t nStrip_wOfs = start + 1; nStrip_wOfs < end;) {
            const uint8_t clustStripN = data_[(nStrip_wOfs) ^ 7];
            nStrip_wOfs += ((uint32_t)clustStripN) * num_bits / 8 + 2;
            stripN += clustStripN;
            // std::cout << (int)nStrip_Ofs << "," << (int)clustStripN << std::endl;
          }
        }
        return stripN;
      }

      ALPAKA_FN_HOST_ACC inline uint8_t packetCode() const { return data_[(offset_ + 2) ^ 7]; }

      ALPAKA_FN_HOST_ACC inline uint16_t cmMedian(uint8_t apvIndex) const {
        uint16_t result = 0;
        result |= data_[(offset_ + 3 + 2 * apvIndex) ^ 7];
        result |= (((data_[(offset_ + 4 + 2 * apvIndex) ^ 7]) << 8) & 0x300);
        return result;
      }

      ALPAKA_FN_HOST_ACC inline const uint8_t* data() const { return data_; }

      ALPAKA_FN_HOST_ACC inline uint32_t offset() const { return offset_; }

    private:
      const uint8_t* data_;
      uint32_t offset_;
      uint16_t length_;
      uint8_t headerLen_;
    };
  }  // namespace fedChannelDetails

  class PortableFEDMover {
  public:
    PortableFEDMover(Queue* queue, uint32_t rawBufferSize, uint32_t fedChannelsNb)
        : buffer(cms::alpakatools::make_host_buffer<uint8_t[]>(*queue, rawBufferSize)),
          mapping(fedChannelsNb, *queue),
          ofs(0),
          chanN(0),
          fedChannelsNb_(fedChannelsNb),
          ofsFedId(sistrip::FED_ID_MAX + 1) {
      // std::cout << "#portBuffSize," << rawBufferSize << std::endl;
      queue_ = queue;
    }

    void fillBuffer(const std::vector<const FEDRawData*>& raw) {
      for (uint16_t fedId = sistrip::FED_ID_MIN; fedId < (sistrip::FED_ID_MAX + 1); fedId++) {
        if (raw[fedId]) {
          std::memcpy(buffer.data() + ofs, raw[fedId]->data(), raw[fedId]->size());
          ofsFedId[fedId] = ofs;
          ofs += raw[fedId]->size();
        }
      }
      // std::cout << "#fillBuffer," << ofs << std::endl;
    }

    void fillMapping(const std::vector<FEDChMetadata>& channelsMeta) {
      for (uint32_t i = 0; i < fedChannelsNb_; i++, chanN++) {
        const auto& channel = channelsMeta[i];
        mapping->detID(i) = channel.detId;
        mapping->fedID(i) = channel.fedId;
        mapping->fedCh(i) = channel.fedCh;
        mapping->fedChOfs(i) = channel.fedChOfs;
        mapping->fedChDataOfsBuf(i) = channel.fedChOfs_wrt_rawFedId + ofsFedId[channel.fedId];
        mapping->readoutMode(i) = channel.fedBufferRO;

        // // Checks that I can rebuilt the FEDChannel
        // bool isNonLite = (mapping->readoutMode(i) == 10 || mapping->readoutMode(i) == 11);
        // auto ch = sistrip::fedChannelDetails::FEDChannel(buffer.data()+mapping->fedChDataOfsBuf(i), channel.fedChOfs, isNonLite);
        // std::cout << "#makeMapping," << i << "," << (int)channel.fedId << "," << (int)channel.fedCh  << "," << ch.length() << "," << (int)channel.fedChOfs  << "," << channel.fedChOfs_wrt_rawFedId  << "," << (int)ch.stripsInCh<8>() << std::endl;
      }
    }

    auto getBuffer() { return buffer; }
    uint32_t getBufferSize() { return ofs; }

    auto getMapping() { return std::move(mapping); }
    uint32_t getChannelsN() { return chanN; }

  private:
    cms::alpakatools::host_buffer<uint8_t[]> buffer;
    SiStripMappingHost mapping;

    uint32_t ofs;
    uint32_t chanN;
    uint32_t fedChannelsNb_;
    std::vector<uint32_t> ofsFedId;

    Queue* queue_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  class SiStripRawToClusterAlgo {
  public:
    SiStripRawToClusterAlgo(const edm::ParameterSet& unpackPar, const edm::ParameterSet& clustPar);
    void prepareUnpackCluster(Queue& queue,
                              const SiStripClusterizerConditionsDetToFedsDevice* rDetToFeds,
                              const SiStripClusterizerConditionsDataDevice* rData,
                              std::unique_ptr<PortableFEDMover> FEDChMover);
    void unpackStrips(Queue& queue);

    std::unique_ptr<SiStripClusterDevice> makeClusters(Queue& queue);
    std::unique_ptr<SiStripDigiDevice> getDigiAmplitudes(Queue& queue);

  private:
    // Clusterizer parameters
    const float channelThreshold_, seedThreshold_, clusterThresholdSquared_;
    const uint8_t maxSequentialHoles_, maxSequentialBad_, maxAdjacentBad_;
    const uint32_t maxClusterSize_;
    const float minGoodCharge_;
    const uint32_t kMaxSeedStrips_;

    // Handles to conditions
    const SiStripClusterizerConditionsDetToFedsDevice* conditions_DetToFeds;
    const SiStripClusterizerConditionsDataDevice* conditions_Data;

    std::unique_ptr<PortableFEDMover> FEDChMover;

    using DeviceBufferU8 = decltype(cms::alpakatools::make_device_buffer<uint8_t[]>(std::declval<Queue>(), 0));
    std::optional<DeviceBufferU8> fedBuffer_d;

    std::optional<SiStripMappingDevice> stripMapping_d;

    using HostBufferU32 = decltype(cms::alpakatools::make_host_buffer<uint32_t>(std::declval<Queue>()));
    std::optional<HostBufferU32> nStrips_h;

    std::optional<HostBufferU32> nSeeds_h;

    int nStripsBytes_ = 0;
    std::unique_ptr<SiStripDigiDevice> digis_d_;
    std::unique_ptr<StripClustersAuxDevice> sClustersAux_d_;

    // Debug functions
    void dumpUnpackedStrips(Queue& queue, SiStripDigiDevice* digis_d);
    void dumpSeeds(Queue& queue, SiStripDigiDevice* digis_d, StripClustersAuxDevice* sClustersAux_d);
    void dumpClusters(Queue& queue, SiStripClusterDevice* clusters_d, SiStripDigiDevice* digis_d);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
