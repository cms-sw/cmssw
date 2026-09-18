#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "EventFilter/Phase2TrackerRawToDigi/interface/SensorHybrid.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include <fstream>
#include <random>

class ClusterToRawProducer : public edm::one::EDProducer<> {
public:
  explicit ClusterToRawProducer(const edm::ParameterSet&);
  ~ClusterToRawProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clusterCollectionToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;

  void insertHexWordAt(unsigned char* data_ptr, size_t word_index, uint32_t hex_word) {
    // Reverse order within each 128-bit (4-word) block
    size_t block_start = (word_index / 4) * 4;
    size_t offset_within_block = word_index % 4;
    size_t remapped_index = block_start + (3 - offset_within_block);
    data_ptr[remapped_index * 4 + 0] = (hex_word >> 0) & 0xFF; // Most significant byte (bits 31-24)
    data_ptr[remapped_index * 4 + 1] = (hex_word >> 8) & 0xFF; // Next byte (bits 23-16)
    data_ptr[remapped_index * 4 + 2] = (hex_word >> 16) & 0xFF;  // Next byte (bits 15-8)
    data_ptr[remapped_index * 4 + 3] = (hex_word >> 24) & 0xFF;  // Least significant byte (bits 7-0)
  }

  void dumpPacket(const unsigned char* data, size_t dataSize);
  void addSLinkHeader(const SLinkRocketHeader_v3& header, std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket);
  void addSLinkTrailer(const SLinkRocketTrailer_v3& trailer, std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket);
  void addTrackerTrailer(const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_BOARD_TYPE_INV>& board_type_inv, std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket);
  void addTrackerHeader(const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_BOARD_TYPE>& board_type,
                        const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_VERSION_MAJOR>& version_major,
                        const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_VERSION_MINOR>& version_minor,
                        const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_MODE>& mode,
                        const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_ED>& ed,
                        const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_BOARD_ID>& board_id,
                        const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_CORE_ID>& core_id,
                        std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket);
  void allocateBytesToBinary(const std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket, 
                           std::vector<unsigned char>& binaryBuffer);
};

ClusterToRawProducer::ClusterToRawProducer(const edm::ParameterSet& iConfig)
    : clusterCollectionToken_(
          consumes<Phase2TrackerCluster1DCollectionNew>(iConfig.getParameter<edm::InputTag>("Phase2Clusters"))),
      cablingMapToken_(esConsumes()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {
  produces<RawDataBuffer>();
}

ClusterToRawProducer::~ClusterToRawProducer() {}

void ClusterToRawProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace Phase2TrackerSpecifications;
  using namespace Phase2DAQFormatSpecification;
  
  // Retrieve TrackerGeometry from EventSetup
  const TrackerGeometry& trackerGeometry = iSetup.getData(trackerGeometryToken_);

  // Retrieve the CablingMap
  const auto& cablingMap = iSetup.getData(cablingMapToken_);

  // get EventID and RunID
  unsigned int eventId_ = iEvent.id().event();

  // Get input clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters_handle;
  iEvent.getByToken(clusterCollectionToken_, clusters_handle);

  // prepare a vector to contain all the slink fragments
  struct SlinkFragment {
    uint64_t source_id;
    std::vector<unsigned char> data;
  };
  std::vector<SlinkFragment> allSlinkFragments;
  allSlinkFragments.reserve((MAX_DTC_ID - MIN_DTC_ID + 1) * (MAX_SLINK_ID + 1));

  size_t totalSize = 0;

  for (int dtc_id = MIN_DTC_ID; dtc_id < MAX_DTC_ID + 1; dtc_id++) {
    for (int slink_id = 0; slink_id < MAX_SLINK_ID + 1; slink_id++) {
      int index_first = slink_id * MODULES_PER_SLINK;
      int index_last = (slink_id + 1) * MODULES_PER_SLINK;
      uint32_t sourceId = slink_id + SLINKS_PER_DTC * (dtc_id - 1) + TRACKER_HEADER;

      /*
       * Preparation of the full fragment for this event. It contains:
       * (1) SLink Header
       * (2) OT Tracker Header
       * (3) Offset Map
       * (4) OT Tracker Payload
       * (5) OT Tracker Trailer
       * (6) SLink Trailer
       */

      std::vector<Word32Bits> daq_packet;
      daq_packet.reserve(SLINK_HEADER_SIZE + HEADER_N_LINES + MODULES_PER_SLINK + DTC_MASK_PROFILE_SIZE);

      /**
       * Configure SLink Rocket Header (Version 3)
       * Following the pattern from TestWriteRawDataBuffer.cc:
       * @see https://github.com/cms-sw/cmssw/blob/2f70a0116630c1586a2a26ffb4b7d256bb8f4b36/DataFormats/FEDRawData/test/TestWriteRawDataBuffer.cc#L36
       */
      uint16_t l1a_types = 1;
      uint8_t l1a_phys = 0xAA;
      uint8_t emu_status = 2;
      SLinkRocketHeader_v3 header (sourceId, 
                                   l1a_types, 
                                   l1a_phys, 
                                   emu_status, 
                                   static_cast<uint64_t>(eventId_));
      addSLinkHeader(header, daq_packet);

      /** Configure defaults for OT Tracker Header
       * @see: https://docs.google.com/spreadsheets/d/1RHZFqeHCoJhRaAfaKEO1Gx6U6c1Y3tRGhL_aSbZQROY/edit?gid=256168213#gid=256168213 for definition.
       **/
      std::bitset<C_NUM_BITS_BOARD_TYPE> board_type(0);                   // 8 bits  (bits 31-24)
      std::bitset<C_NUM_BITS_BOARD_TYPE_INV> board_type_inv(0);           // 8 bits  (bits 31-24)
      std::bitset<C_NUM_BITS_VERSION_MAJOR> version_major(VERSION_MAJOR_V1_0); // 5 bits  (bits 23-19)
      std::bitset<C_NUM_BITS_VERSION_MINOR> version_minor(VERSION_MINOR_V1_0); // 3 bits  (bits 18-16)
      std::bitset<C_NUM_BITS_MODE> mode(0);                               // 3 bits  (bits 15-13)
      std::bitset<C_NUM_BITS_ED> ed(0);                                   // 1 bit   (bit 12)
      std::bitset<C_NUM_BITS_BOARD_ID> board_id(0);                       // 8 bits  (bits 11-4)
      std::bitset<C_NUM_BITS_CORE_ID> core_id(0);                         // 4 bits  (bits 3-0)
      bool board_type_set = false;

      /** Firmware Accurate Offset Counter **/
      unsigned int offset_in_32b_words = 0;

      /** Firmware Accurate OT Tracker Stream Components
       * (1) Offset Map
       * (2) OT Payload Section
       */
      std::vector<Word32Bits> offset_map(CICs_PER_SLINK / 2, Word32Bits(0));
      std::vector<Word32Bits> payload;

      /** Iterate Over Modules for this Specific FED **/
      for (int module_id = index_first; module_id < index_last; module_id++) {
        const unsigned int module_id_within_slink = module_id - index_first;
        DTCELinkId cms_link_id = DTCELinkId(dtc_id, module_id, 0);
        try {
          auto link_to_det_association = cablingMap.dtcELinkIdToDetId(cms_link_id);
          const DetId& det_id = link_to_det_association->second;

          if (board_type_set == false) {
            TrackerGeometry::ModuleType moduleType = trackerGeometry.getDetectorType(det_id);
            if (moduleType == TrackerGeometry::ModuleType::Ph2PSS || moduleType == TrackerGeometry::ModuleType::Ph2PSP) {
              board_type = DTC_HEADER_OT_PS;
              board_type_inv = DTC_HEADER_OT_PS_INV;
            } else if (moduleType == TrackerGeometry::ModuleType::Ph2SS) {
              board_type = DTC_HEADER_OT_2S;
              board_type_inv = DTC_HEADER_OT_2S_INV;
            } else {
              throw cms::Exception("That's Impossible.");
            }
            core_id = module_id / MODULES_PER_SLINK;
            board_id = dtc_id;
            board_type_set = true;
          }
          
          edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_1_cluster_collection =
              clusters_handle->find(det_id + 1);
          edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_2_cluster_collection =
              clusters_handle->find(det_id + 2);
          const edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator nullIter = clusters_handle->end();

          // sensor_1_cic_0 and sensor_2_cic_0 form a single output daq channel.
          SensorHybrid hybrid_1(
              det_id, sensor_1_cluster_collection, sensor_2_cluster_collection, nullIter, false, trackerGeometry, eventId_);

          // // sensor_1_cic_1 and sensor_2_cic_1 form a single output daq channel.
          SensorHybrid hybrid_2(
              det_id, sensor_1_cluster_collection, sensor_2_cluster_collection, nullIter, true, trackerGeometry, eventId_);

          // sensor_2 is always isUpper == 1 for 2S.
          // sensor_2 is always isLower == 0 for 2S.

          // Figure Out Offsets
          offset_in_32b_words += hybrid_1.get_payload_size();
          uint16_t hybrid_1_offset = offset_in_32b_words;

          offset_in_32b_words += hybrid_2.get_payload_size();
          uint16_t hybrid_2_offset = offset_in_32b_words;

          // 24 is PSS, 23 is PSP, 26 is SS-SS
          uint32_t combined_offsets = (static_cast<uint32_t>(hybrid_1_offset) << 16) | hybrid_2_offset;
          offset_map[module_id_within_slink] = Word32Bits(combined_offsets);

          // Figure out Payload
          hybrid_1.get_payload(payload);
          hybrid_2.get_payload(payload);
        } catch (const cms::Exception& e) {
          // exception here means that the link is not connected to a detector
          uint32_t eventID = CIC_CONSTANT_EVENT_ID & L1ID_MAX_VALUE;  // eventId_ (9 bits)
          uint32_t channelErrors = 0;                    // 9 bits for errors, all set to 0
          uint32_t numClusters = 0;                      // no clusters here.

          // Build the channel header
          uint32_t header_ = (eventID << (N_BITS_PER_WORD - L1ID_BITS)) |
                             (channelErrors << (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) |
                             (numClusters << (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) |
                             (numClusters);

          uint16_t hybrid_1_offset = offset_in_32b_words;
          offset_in_32b_words += 1;

          uint16_t hybrid_2_offset = offset_in_32b_words;
          offset_in_32b_words += 1;

          uint32_t combined_offsets = (static_cast<uint32_t>(hybrid_2_offset) << 16) | hybrid_1_offset;
          offset_map[module_id_within_slink] = Word32Bits(combined_offsets);

          // Push the header into the payload
          payload.push_back(Word32Bits(header_));
          payload.push_back(Word32Bits(header_));

          // continue;
        }
      }

      /** 
       * Configure and Add Tracker Header
       */
      addTrackerHeader(board_type, version_major, version_minor, mode, ed, board_id, core_id, daq_packet);

      // Add the offset map to the slink_daq_stream
      for (std::size_t i = 0; i < offset_map.size(); i++) {
        daq_packet.push_back(offset_map[i]);
      }

      // Tracker Mask
      for (int i = 0; i < RESERVED_N_LINES; ++i) {
        daq_packet.push_back(Word32Bits(0));
      }
//       daq_packet.push_back(0x0);
//       daq_packet.push_back(0x0);

      daq_packet.reserve(daq_packet.size() + payload.size());

      // Pad payload to 128b  // is it needed ? FIXME
      size_t payload_data_bytes = payload.size() * N_BYTES_PER_WORD;
      size_t padded_payload_bytes = (payload_data_bytes + 15) & ~static_cast<size_t>(15);  // pad to 16-byte boundary, required by RawDataBuffer
      size_t padding_bytes = padded_payload_bytes - payload_data_bytes;
      size_t padding_words = padding_bytes / N_BYTES_PER_WORD;
      for (size_t i = 0; i < padding_words; ++i) {
        payload.push_back(Word32Bits(0));
      }

      // Add the payload to the slink_daq_stream
      for (std::size_t i = 0; i < payload.size(); i++) {
        daq_packet.push_back(payload[i]);
      }

      // Pad to nearest 128b boundary.
      while (daq_packet.size() % 4 != 0) {
        daq_packet.push_back(Word32Bits(0));
      }

      /** 
       * Configure and Add Tracker Trailer
       */
      addTrackerTrailer(board_type_inv, daq_packet);

      /** 
       * Configure and Add SLink Trailer
       * Following the pattern from TestWriteRawDataBuffer.cc:
       * @see: https://github.com/cms-sw/cmssw/blob/2f70a0116630c1586a2a26ffb4b7d256bb8f4b36/DataFormats/FEDRawData/test/TestWriteRawDataBuffer.cc#L36
       */
      uint16_t slt_status = 0;               // SLink trailer status (0 = OK)
      uint16_t crc = 0;                      // CRC (not computed, set to 0)
      uint32_t orbit_id = 0;                 // Orbit ID (test value from reference)
      uint16_t bx_id = 0;                    // Bunch crossing ID (test value from reference)
      uint32_t fragment_size_words = daq_packet.size() + 4;  // Total fragment size in 32-bit words (including trailer)
      SLinkRocketTrailer_v3 trailer(slt_status, crc, orbit_id, bx_id, fragment_size_words, 0);
      addSLinkTrailer(trailer, daq_packet);

      /** Convert DAQ Packet to Raw Bytes, As They Appear in Real Binaries **/
      std::vector<unsigned char> slink_daq_stream;
      allocateBytesToBinary(daq_packet, slink_daq_stream);

      /** Update Total Size and Store Fragment **/
      const size_t fragment_bytes = slink_daq_stream.size() * N_BYTES_PER_WORD;
      totalSize += fragment_bytes;  
      allSlinkFragments.push_back({sourceId, std::move(slink_daq_stream)});
    }
  }

  // Create RawDataBuffer to store the output
  auto rawDataBuffer = std::make_unique<RawDataBuffer>(totalSize);

  for (auto& fragment : allSlinkFragments) {
    rawDataBuffer->addSource(fragment.source_id, fragment.data.data(), fragment.data.size());
  }

  iEvent.put(std::move(rawDataBuffer));
}

/**
 * @brief Dumps the entire DAQ Packet, in hexdump -C view.
 * @param data Pointer to the raw data buffer
 * @param dataSize Size of the data buffer in bytes
 * @return void
 */
void ClusterToRawProducer::dumpPacket(const unsigned char* data, size_t dataSize) {
  for (size_t l16byteslineID = 0; l16byteslineID < (dataSize + 15) / 16; l16byteslineID++) {
    for (size_t byte_within_line = 0; byte_within_line < 16; byte_within_line++) {
      size_t index = l16byteslineID * 16 + byte_within_line;
      if (index >= dataSize) break;  // Stop if we've printed all bytes
      printf("%02X ", (unsigned int)data[index]);
    }
    printf("\n");
  }
}

/**
 * @brief Adds an SLinkRocketHeader_v3 to a DAQ packet vector
 * @param header The SLink header to add
 * @param daqPacket The vector to append the header words to
 */
void ClusterToRawProducer::addSLinkHeader(const SLinkRocketHeader_v3& header, std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket) {

  const unsigned char* header_bytes = reinterpret_cast<const unsigned char*>(&header);
  size_t header_size = sizeof(SLinkRocketHeader_v3);
  
  // Add each word (assuming header is multiple of 4 bytes)
  for (size_t i = 0; i < header_size; i += 4) {
      uint32_t word = 0;
      for (size_t j = 0; j < 4 && (i + j) < header_size; j++) {
          word |= (header_bytes[i + j] << (j * 8));
      }
      daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(word));
  }
}

/**
 * @brief Adds an SLinkRocketTrailer_v3 to a DAQ packet vector
 * @param trailer The SLink trailer to add
 * @param daqPacket The vector to append the trailer words to
 */
void ClusterToRawProducer::addSLinkTrailer(const SLinkRocketTrailer_v3& trailer, std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket) {
    const unsigned char* trailer_bytes = reinterpret_cast<const unsigned char*>(&trailer);
    size_t trailer_size = sizeof(SLinkRocketTrailer_v3);
    
    // Add each word (assuming trailer is multiple of 4 bytes)
    for (size_t i = 0; i < trailer_size; i += 4) {
        uint32_t word = 0;
        for (size_t j = 0; j < 4 && (i + j) < trailer_size; j++) {
            word |= (trailer_bytes[i + j] << (j * 8));
        }
        daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(word));
    }
}

/**
 * @brief Adds the OT Tracker Trailer to the DAQ packet
 * @param board_type_inv The inverted board type (placed in bits 31-24)
 * @param daqPacket The vector to append the trailer words to
 */
void ClusterToRawProducer::addTrackerTrailer(const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_BOARD_TYPE_INV>& board_type_inv, 
                                              std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket) {
    uint32_t last_word = board_type_inv.to_ulong() << 24;  // Put it in bits 31-24
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(last_word));
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(0x0));
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(0x0));
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(0x0));
}

/**
 * @brief Adds the OT Tracker Header to the DAQ packet
 * @param board_type 8-bit board type (bits 31-24)
 * @param version_major 5-bit major version (bits 23-19)
 * @param version_minor 3-bit minor version (bits 18-16)
 * @param mode 3-bit mode (bits 15-13)
 * @param ed 1-bit ED (bit 12)
 * @param board_id 8-bit board ID (bits 11-4)
 * @param core_id 4-bit core ID (bits 3-0)
 * @param daqPacket The vector to append the header words to
 * @see: https://docs.google.com/spreadsheets/d/1RHZFqeHCoJhRaAfaKEO1Gx6U6c1Y3tRGhL_aSbZQROY/edit?gid=256168213#gid=256168213 for definition.
 */
void ClusterToRawProducer::addTrackerHeader(const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_BOARD_TYPE>& board_type,
                                            const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_VERSION_MAJOR>& version_major,
                                            const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_VERSION_MINOR>& version_minor,
                                            const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_MODE>& mode,
                                            const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_ED>& ed,
                                            const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_BOARD_ID>& board_id,
                                            const std::bitset<Phase2DAQFormatSpecification::C_NUM_BITS_CORE_ID>& core_id,
                                            std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket) {
    // Build the first word from all the bit fields
    Phase2DAQFormatSpecification::Word32Bits first_word(board_type.to_string() + 
                          version_major.to_string() + 
                          version_minor.to_string() + 
                          mode.to_string() + 
                          ed.to_string() + 
                          board_id.to_string() + 
                          core_id.to_string());
    
    daqPacket.push_back(first_word);
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(0x0));
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(0x0));
    daqPacket.push_back(Phase2DAQFormatSpecification::Word32Bits(0x0));
}

/**
 * @brief Converts a DAQ packet vector to raw binary bytes with proper byte ordering
 * @param daqPacket The input DAQ packet vector of 32-bit words
 * @param binaryBuffer The output binary buffer to fill with bytes
 * This function:
 * 1. Resizes the binary buffer to hold all bytes from daqPacket
 * 2. Handles SLink Header (first 4 words) and Trailer (last 4 words) specially
 * 3. Applies 128-bit block reversal to the middle payload words
 * 4. Adds padding to match the captured binary format
 */
void ClusterToRawProducer::allocateBytesToBinary(const std::vector<Phase2DAQFormatSpecification::Word32Bits>& daqPacket, 
                                                   std::vector<unsigned char>& binaryBuffer) {
    // Calculate size and padding
    size_t size_in_bytes = daqPacket.size() * Phase2DAQFormatSpecification::N_BYTES_PER_WORD;
    size_t padding = (Phase2DAQFormatSpecification::N_BYTES_PER_DTH_BINARY_WORD - (size_in_bytes % Phase2DAQFormatSpecification::N_BYTES_PER_DTH_BINARY_WORD)) % Phase2DAQFormatSpecification::N_BYTES_PER_DTH_BINARY_WORD;
    
    // Resize and initialize binary buffer
    binaryBuffer.resize(size_in_bytes + padding, Phase2DAQFormatSpecification::N_BYTES_PER_DTH_BINARY_WORD);
    unsigned char* data_ptr = binaryBuffer.data();
    
    // Write each 32-bit word to the raw buffer
    for (size_t word_index = 0; word_index < daqPacket.size(); ++word_index) {
        // Handle both SLink Header (first 4 words) and SLink Trailer (last 4 words)
        if (word_index < 4 || word_index >= daqPacket.size() - 4) {
            // SLink Header/Trailer: write in little-endian order
            uint32_t word = daqPacket[word_index].to_ulong();
            data_ptr[word_index * 4 + 0] = (word >> 0) & 0xFF;
            data_ptr[word_index * 4 + 1] = (word >> 8) & 0xFF;
            data_ptr[word_index * 4 + 2] = (word >> 16) & 0xFF;
            data_ptr[word_index * 4 + 3] = (word >> 24) & 0xFF;
        } else {
            // Payload: apply 128-bit block reversal
            insertHexWordAt(data_ptr, word_index, daqPacket[word_index].to_ulong());
        }
    }
}

DEFINE_FWK_MODULE(ClusterToRawProducer);
