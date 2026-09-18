#ifndef Phase2DAQFormatSpecification_H
#define Phase2DAQFormatSpecification_H

namespace Phase2DAQFormatSpecification {

  static constexpr int SLINKS_PER_DTC = 16;
  static constexpr uint32_t CHIP_HEADER_MAGIC = 0xE;
  static constexpr uint32_t HEADER_TRAILER_PATTERN = 0xFFFFFFFF;
  static constexpr int HEADER_TRAILER_LINES = 4;
  static constexpr int BITS_PER_WORD = 32;
  static constexpr int BITS_PER_CHUNK = 128;
  static constexpr int BYTES_PER_WORD = 4;
  static constexpr int CHIPS_PER_MODULE = 4;

  static constexpr int HITMAP_SIZE = 16;
  static constexpr int HITMAP_ROW = 4;
  static constexpr int HITMAP_COL = 4;

  static constexpr int AURORA_BLOCK_BODY_BITS = 64;   // unscrambled aurora block body
  static constexpr int AURORA_BLOCK_TOTAL_BITS = 66;  // aurora block body + 2 bits sync header
  static constexpr int AURORA_SYNC_HEADER_BITS = 2;   // aurora synching header (2 bits)
  static constexpr int AURORA_NS_BITS = 1;
  static constexpr int AURORA_CHIP_ID_BITS = 2;       // 2 bits for multi-chip module chip indexing
  static constexpr int AURORA_EOS_MIN_TRAILING_ZEROS = 6;
  static constexpr int AURORA_SERVICE_BLOCK_INTERVAL_DEFAULT = 50;  // ND service block interval
  static constexpr int AURORA_SERVICE_BLOCK_INTERVAL_MIN = 1;
  static constexpr int AURORA_SERVICE_BLOCK_INTERVAL_MAX = 256;
  static constexpr int AURORA_EVENTS_PER_STREAM_DEFAULT = 16;       // NE events unaligned stream group
  static constexpr int AURORA_EVENTS_PER_STREAM_MIN = 1;
  static constexpr int AURORA_EVENTS_PER_STREAM_MAX = 64;

};  // namespace Phase2DAQFormatSpecification

#endif
