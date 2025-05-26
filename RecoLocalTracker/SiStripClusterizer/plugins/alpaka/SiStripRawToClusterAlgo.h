#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h

#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"
#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"

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
}  // namespace sistripclusterizer

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace sistripclusterizer;

  using StripClustersAuxHost = ::sistripclusterizer::StripClustersAuxHost;
  using StripClustersAuxDevice = PortableCollection<StripClustersAuxSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

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
  struct Det{
    uint32_t detId=0;
    uint16_t fedId=0;
    uint8_t fedCh=0;
    uint8_t apvPair=0;
  };

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
    class FEDChannel {
    public:

      ALPAKA_FN_HOST_ACC inline FEDChannel(const uint8_t* data, uint32_t offset, bool isNonLite)
        : data_(data), offset_(offset), headerLen_(isNonLite ? 7 : 2) {
          length_ = (data_[(offset_) ^ 7] + (data_[(offset_ + 1) ^ 7] << 8));
        }

      ALPAKA_FN_HOST_ACC inline uint16_t length() const {
        return length_;
      }

      template <uint_fast8_t num_bits>
      ALPAKA_FN_HOST_ACC inline uint16_t stripsInCh() {
        const bool emptyCh = (headerLen_+2) >= (length_);
        const uint16_t ofsWordUnit = num_bits/8; 
        const uint16_t start = offset_ + headerLen_;
        const uint16_t end = offset_ + length_;
        if (!emptyCh){
          for(uint16_t nStrip_Ofs = start+1; nStrip_Ofs<end;) {
            const uint8_t clustStripN = data_[(nStrip_Ofs) ^ 7];
            nStrip_Ofs += clustStripN*ofsWordUnit + 2;
            stripN_ += clustStripN;
            // std::cout << (int)nStrip_Ofs << "," << (int)clustStripN << std::endl;
          }
        }
        return stripN_;
      }

      ALPAKA_FN_HOST_ACC inline uint8_t packetCode() const {
        return data_[(offset_ + 2) ^ 7];
      }

      ALPAKA_FN_HOST_ACC inline uint16_t cmMedian(uint8_t apvIndex) const {
        uint16_t result = 0;
        result |= data_[(offset_ + 3 + 2 * apvIndex) ^ 7];
        result |= (((data_[(offset_ + 4 + 2 * apvIndex) ^ 7]) << 8) & 0x300);
        return result;
      }

      ALPAKA_FN_HOST_ACC inline const uint8_t* data() const {
        return data_;
      }

      ALPAKA_FN_HOST_ACC inline uint32_t offset() const {
        return offset_;
      }

    private:
      const uint8_t* data_;
      uint32_t offset_;
      uint16_t length_;
      uint8_t headerLen_;

      //
      uint16_t stripN_=0;
    };
  }

  class PortableFEDMover {
    public:
    PortableFEDMover(Queue* queue, uint32_t rawBufferSize, uint32_t fedChannelsNb)
      : buffer(cms::alpakatools::make_host_buffer<uint8_t[]>(*queue, rawBufferSize)),
        mapping(fedChannelsNb, *queue),
        ofs(0),
        chanN(0),
        fedChannelsNb_(fedChannelsNb),
        ofsFedId(sistrip::FED_ID_MAX+1) {
          // std::cout << "#portBuffSize," << rawBufferSize << std::endl;
          queue_ = queue;
    }

    void fillBuffer(const std::vector<const FEDRawData*>& raw) {
      for(uint16_t fedId = sistrip::FED_ID_MIN; fedId < (sistrip::FED_ID_MAX+1); fedId++){
        if (raw[fedId]) {
          std::memcpy(buffer.data() + ofs, raw[fedId]->data(), raw[fedId]->size());
          ofsFedId[fedId] = ofs;
          ofs += raw[fedId]->size();
        }
      }
      // std::cout << "#fillBuffer," << ofs << std::endl;
    }

    void fillMapping(const std::vector<FEDChMetadata>& channelsMeta) {
      for (uint32_t i=0; i< fedChannelsNb_; i++, chanN++) {
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

    auto getBuffer(){ return buffer; }
    uint32_t getBufferSize(){ return ofs; }
    
    auto getMapping(){ return std::move(mapping); }
    uint32_t getChannelsN(){ return chanN; }

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
    void prepareUnpackCluster(Queue& queue, const SiStripClusterizerConditionsDetToFedsDevice* rDetToFeds, const SiStripClusterizerConditionsDataDevice* rData, std::unique_ptr<PortableFEDMover> FEDChMover);
    void unpackStrips2(Queue& queue);
    void prefixScan_new(Queue& queue);
    std::unique_ptr<SiStripClustersDevice> makeClusters2(Queue& queue);
    std::unique_ptr<SiStripDigiDevice> getDigiAmplitudes(Queue& queue);

    //
    private:
    // Unpacker parameters
    const bool isLegacyUnpacker_;
    const FEDLegacyReadoutMode legacyUnpackerROmode_ = READOUT_MODE_LEGACY_INVALID;
    
    std::unique_ptr<PortableFEDMover> FEDChMover;
    using DeviceBufferU8 = decltype(cms::alpakatools::make_device_buffer<uint8_t[]>(std::declval<Queue>(), 0));
    std::optional<DeviceBufferU8> fedBuffer_d;
    
    std::optional<SiStripMappingDevice> stripMapping_d;

    const SiStripClusterizerConditionsDetToFedsDevice* conditions_DetToFeds;
    const SiStripClusterizerConditionsDataDevice* conditions_Data;

    using HostBufferU32 = decltype(cms::alpakatools::make_host_buffer<uint32_t>(std::declval<Queue>()));
    std::optional<HostBufferU32> nStrips_h;
    
    std::optional<HostBufferU32> nSeeds_h;



    // Clusterizer parameters
    const float channelThreshold_, seedThreshold_, clusterThresholdSquared_;
    const uint8_t maxSequentialHoles_, maxSequentialBad_, maxAdjacentBad_;
    const uint32_t maxClusterSize_;
    const float minGoodCharge_;
    const uint32_t kMaxSeedStrips_;

    int nStripsBytes_ = 0;
    std::unique_ptr<SiStripDigiDevice> digis_d_;
    std::unique_ptr<StripClustersAuxDevice> sClustersAux_d_;

    void dumpUnpackedStrips(Queue& queue, SiStripDigiDevice* digis_d);
    void dumpSeeds(Queue& queue, SiStripDigiDevice* digis_d, StripClustersAuxDevice* sClustersAux_d);
    void dumpClusters(Queue& queue, SiStripClustersDevice* clusters_d, SiStripDigiDevice* digis_d);
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif  // RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterAlgo_h
