#ifndef DataFormats_Phase2TrackerDigi_Phase2ITChipBitStream_H
#define DataFormats_Phase2TrackerDigi_Phase2ITChipBitStream_H
#include <cstdint>
#include <vector>

// Encoded bit stream output from each chips
class Phase2ITChipBitStream {
public:
  Phase2ITChipBitStream(int rocid, std::vector<uint8_t> bytes, uint32_t nBits)
      : rocid_(rocid), bytes_(std::move(bytes)), nBits_(nBits) {}

  Phase2ITChipBitStream() : rocid_(-1), nBits_(0) {}

  int get_rocid() const { return rocid_; }

  const std::vector<uint8_t>& bytes() const { return bytes_; }
  uint32_t nBits() const { return nBits_; }
  bool bit(uint32_t i) const { return (bytes_[i / 8] >> (7 - i % 8)) & 1; }

  // Rebuilding the bit vector only if called
  std::vector<bool> get_bitstream() const {
    std::vector<bool> out(nBits_);
    for (uint32_t i = 0; i < nBits_; ++i)
      out[i] = bit(i);
    return out;
  }

  const bool operator<(const Phase2ITChipBitStream& other) { return rocid_ < other.rocid_; }

private:
  int rocid_;                   // Chip index
  std::vector<uint8_t> bytes_;  // Chip bit stream
  uint32_t nBits_;              // Significant bits in bytes_
};
#endif  // DataFormats_Phase2TrackerDigi_Phase2ITChipBitStream_H
