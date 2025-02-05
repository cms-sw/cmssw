#ifndef EventFilter_Utilities_DTHHeaders_h
#define EventFilter_Utilities_DTHHeaders_h

#include <array>
#include <cstddef>
#include <cstdint>

/*
 * DTH Orbit header, event fragment trailer and SlinkRocket Header and Trailer accompanying
 * slink payload. Format that is sent is is low-endian for multi-byte fields
 *
 * Version 1 DTH and Version 3 SLinkRocket
 *
 * */

namespace evf {
  constexpr uint32_t SLR_MAX_EVENT_LEN = (1 << 20) - 1;
  constexpr std::array<uint8_t, 2> DTHOrbitMarker{{0x48, 0x4f}}; //HO
  constexpr std::array<uint8_t, 2> DTHFragmentTrailerMarker{{0x54, 0x46}}; //TF
  constexpr uint32_t DTH_WORD_NUM_BYTES_SHIFT = 4;
  constexpr uint32_t SLR_WORD_NUM_BYTES_SHIFT = 4;


  class DTHOrbitHeader_v1 {
  public:

    DTHOrbitHeader_v1(uint32_t source_id,
                      uint32_t run_number,
                      uint32_t orbit_number,
                      uint16_t event_count,
                      uint32_t packed_word_count,
                      uint32_t flags,
                      uint32_t crc)
        : source_id_(source_id),
          run_number_(run_number),
          orbit_number_(orbit_number),
          event_count_(event_count),
          packed_word_count_(packed_word_count),
          crc32c_(crc) {
            flags_.all_ = flags;
    }


    uint16_t version() const { return version_; }
    uint32_t sourceID() const { return source_id_; }
    uint32_t runNumber() const { return run_number_; }
    uint32_t orbitNumber() const { return orbit_number_; }
    uint16_t eventCount() const { return event_count_; }
    uint32_t packed_word_count() const { return packed_word_count_; }
    uint32_t flags() const { return flags_.all_; }
    uint32_t crc() const { return crc32c_; }

    uint64_t totalSize() const { return uint64_t(packed_word_count_) << DTH_WORD_NUM_BYTES_SHIFT; }
    uint64_t payloadSizeBytes() const { return totalSize() - sizeof(DTHOrbitHeader_v1); }
    uint64_t headerSize() const { return sizeof(DTHOrbitHeader_v1); }
    const void* payload() const { return (uint8_t*)this + sizeof(DTHOrbitHeader_v1); }
    bool verifyMarker() const {
      for (size_t i = 0; i < DTHOrbitMarker.size(); i++) {
        if (marker_[i] != DTHOrbitMarker[i])
          return false;
      }
      return true;
    }

    bool verifyChecksum() const;

  private:
    std::array<uint8_t, 2> marker_ = DTHOrbitMarker;
    uint16_t version_ = 1; //bytes: 01 00
    uint32_t source_id_;
    uint32_t run_number_;
    uint32_t orbit_number_;
    uint32_t event_count_:12,
	     res_:20;
    uint32_t packed_word_count_; //Total size encoded in multiples of 128 bits (16 bytes)
    union {
	struct {
	uint32_t error_flag_:1,
		res_flags_:31;
	} bits_;
	uint32_t all_;
    } flags_;
    uint32_t crc32c_;
  };


  class DTHFragmentTrailer_v1 {
  public:

    DTHFragmentTrailer_v1(
                          uint16_t flags,
                          uint32_t payload_word_count,
                          uint64_t event_id,
                          uint16_t crc)
        : payload_word_count_(payload_word_count),
          event_id_(event_id),
          crc_(crc) {
      flags_.all_ = flags;
    }


    uint64_t eventID() const { return event_id_; }
    uint32_t payloadWordCount() const { return payload_word_count_; }
    uint64_t payloadSizeBytes() const { return uint64_t(payload_word_count_) << DTH_WORD_NUM_BYTES_SHIFT; }
    uint16_t flags() const { return flags_.all_; }
    uint16_t crc() const { return crc_; }
    const void* payload() const { return (uint8_t*)this - payloadSizeBytes(); }
    bool verifyMarker() const {
      for (size_t i = 0; i < DTHFragmentTrailerMarker.size(); i++) {
        if (marker_[i] != DTHFragmentTrailerMarker[i])
          return false;
      }
      return true;
    }

  private:
    std::array<uint8_t, 2> marker_ = DTHFragmentTrailerMarker;
    union {
      struct {
        uint16_t fed_crc_error_:1,
	         slink_crc_error_:1,
	         source_id_error_:1,
		 fragment_cut_:1,
		 event_id_sync_error_:1,
		 fragment_timout_:1,
		 fragment_length_error_:1,
		 res_:9;
      } bits_;
      uint16_t all_;
    } flags_;
    uint32_t payload_word_count_;  // Fragment size is encoded in multiples of 128 bits (16 bytes). I.e needs to be shifted by 4
    uint64_t event_id_:44,
             res_:4,
             crc_:16;
  };


  //SLinkExpress classes

  //begin and end event markers
  constexpr uint8_t SLR_BOE = 0x55;
  constexpr uint8_t SLR_EOE = 0xaa;

  //minimal SLinkRocket version overlay
  class SLinkRocketHeader_version {
  public:
    SLinkRocketHeader_version(uint8_t version, uint8_t res = 0) : res_(res), version_(version) {}
    uint8_t version() const { return version_; }
    bool verifyMarker() const { return boe_ == SLR_BOE; }

  private:
    uint8_t res_:4,
	    version_:4;
    uint8_t boe_ = SLR_BOE;

  };

  class SLinkRocketHeader_v3 {
  public:

    SLinkRocketHeader_v3(uint32_t source_id,
                         uint16_t l1a_types,
                         uint8_t l1a_phys,
                         uint8_t emu_status,
                         uint64_t event_id)
        : source_id_(source_id),
          l1a_types_(l1a_types),
          phys_type_(l1a_phys),
          emu_status_(emu_status),
          event_id_(event_id) {}


    uint32_t sourceID() const { return source_id_; }
    uint16_t l1aTypes() const { return l1a_types_; }
    uint8_t l1aPhysType() const { return phys_type_; }
    uint8_t emuStatus() const { return emu_status_; }
    uint64_t globalEventID() const { return event_id_;}
    uint8_t version() const { return version_; }
    bool verifyMarker() const { return boe_ == SLR_BOE; }

  private:
    uint32_t source_id_;
    uint16_t l1a_types_;
    uint8_t  phys_type_;
    uint8_t  emu_status_:2,
	     res1_:6;
    uint64_t event_id_:44,
	     res2_:8,
	     version_:4 = 3,
	     boe_:8 = SLR_BOE;
  };


  class SLinkRocketTrailer_v3 {
  public:

    SLinkRocketTrailer_v3(
        uint16_t status,
        uint16_t crc,
        uint32_t orbit_id,
        uint16_t bx_id,
        uint32_t evtlen_word_count,
        uint16_t daq_crc)
        : crc_(crc),
          orbit_id_(orbit_id),
          bx_id_(bx_id),
          event_length_wcount_(evtlen_word_count),
          daq_crc_(daq_crc) {
     status_.all_ = status;
   }


    uint16_t status() const { return status_.all_; }
    uint16_t crc() const { return crc_; }
    uint32_t orbitID() const { return orbit_id_; }
    uint16_t bxID() const { return bx_id_; }
    uint32_t eventLenBytes() const { return uint32_t(event_length_wcount_) << SLR_WORD_NUM_BYTES_SHIFT; }
    uint16_t daqCRC() const { return daq_crc_; }
    bool verifyMarker() const { return eoe_ == SLR_EOE; }

  private:

    union {
      struct {
        uint16_t fed_crc_error_:1,     /* FED CRC error was detected by DTH and corrected */
                 slink_crc_error_:1,   /* Set when the slink receviver finds a mistmatch between calculated crc and daq_crc. It should never happen */
                 source_id_error_:1,   /* SOURCE_ID is not expected */
                 sync_lost_error_:1,   /* Sync lost detected by DTH */
                 fragment_cut_:1,      /* Fragment was cut */
                 res_:11;
      } bits_;
      uint16_t all_;
    } status_;
    uint16_t crc_;                         /* CRC filled by the FED */
    uint32_t orbit_id_;
    uint32_t bx_id_:12,
             event_length_wcount_:20;      /* Length is encoded in multiples of 128 bits (16 bytes). I.e needs to be shifter by 4 */
    uint32_t reserved_:8,
             daq_crc_:16,                  /* CRC filled by the slink sender */
             eoe_:8 = SLR_EOE;

  };

}  // namespace evf

#endif
