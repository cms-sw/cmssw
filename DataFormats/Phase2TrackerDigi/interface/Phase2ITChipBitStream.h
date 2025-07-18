#ifndef DataFormats_Phase2TrackerDigi_Phase2ITChipBitStream_H
#define DataFormats_Phase2TrackerDigi_Phase2ITChipBitStream_H
#include <vector>

class Phase2ITChipBitStream {
  // Encoded bit stream output from chips
public:
  Phase2ITChipBitStream(int rocid, const std::vector<bool>& bitstream) {
    rocid_ = rocid;
    bitstream_ = bitstream;
  }

  Phase2ITChipBitStream() { rocid_ = -1; }

  int get_rocid() const { return rocid_; }

  const std::vector<bool>& get_bitstream() const { return bitstream_; }

  const bool operator<(const Phase2ITChipBitStream& other) { return rocid_ < other.rocid_; }

private:
  int rocid_;                    // Chip index
  std::vector<bool> bitstream_;  // Chip bit stream output
};
#endif  // DataFormats_Phase2TrackerDigi_Phase2ITChipBitStream_H
