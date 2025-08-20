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
  constexpr std::array<uint8_t, 2> DTHOrbitMarker{{0x48, 0x4f}};            //HO
  constexpr std::array<uint8_t, 2> DTHFragmentTrailerMarker{{0x54, 0x46}};  //TF
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
    uint16_t version_ = 1;  //bytes: 01 00
    uint32_t source_id_;
    uint32_t run_number_;
    uint32_t orbit_number_;
    uint32_t event_count_ : 12, res_ : 20;
    uint32_t packed_word_count_;  //Total size encoded in multiples of 128 bits (16 bytes)
    union {
      struct {
        uint32_t error_flag_ : 1, res_flags_ : 31;
      } bits_;
      uint32_t all_;
    } flags_;
    uint32_t crc32c_;
  };

  class DTHFragmentTrailer_v1 {
  public:
    DTHFragmentTrailer_v1(uint16_t flags, uint32_t payload_word_count, uint64_t event_id, uint16_t crc)
        : payload_word_count_(payload_word_count), event_id_(event_id), crc_(crc) {
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
        uint16_t fed_crc_error_ : 1, slink_crc_error_ : 1, source_id_error_ : 1, fragment_cut_ : 1,
            event_id_sync_error_ : 1, fragment_timout_ : 1, fragment_length_error_ : 1, res_ : 9;
      } bits_;
      uint16_t all_;
    } flags_;
    uint32_t
        payload_word_count_;  // Fragment size is encoded in multiples of 128 bits (16 bytes). I.e needs to be shifted by 4
    uint64_t event_id_ : 44, res_ : 4, crc_ : 16;
  };

}  // namespace evf

#endif
