#ifndef DataFormats_Phase2TrackerDigi_ROCBitStream_H
#define DataFormats_Phase2TrackerDigi_ROCBitStream_H
#include <vector>

class ROCBitStream {
public:
  ROCBitStream(int rocid, const std::vector<bool>& bitstream) {
    rocid_ = rocid;
    bitstream_ = bitstream;
  }

  ROCBitStream() { rocid_ = -1; }

  int get_rocid() const { return rocid_; }

  const std::vector<bool>& get_bitstream() const { return bitstream_; }

  const bool operator<(const ROCBitStream& other) { return rocid_ < other.rocid_; }

private:
  int rocid_;
  std::vector<bool> bitstream_;
};
#endif  // DataFormats_Phase2TrackerDigi_ROCBitStream_H
