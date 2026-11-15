#ifndef DataFormats_Phase2TrackerDigi_Phase2ITBitBuffer_H
#define DataFormats_Phase2TrackerDigi_Phase2ITBitBuffer_H

#include <cstdint>
#include <vector>

// Writing counterpart of Phase2ITBitReader
// Instead of casting new objects in each step, create the buffer
class Phase2ITBitBuffer {
public:
  void push(bool b) {
    if ((nBits_ & 7) == 0)
      bytes_.push_back(0);
    if (b)
      bytes_.back() |= uint8_t(1) << (7 - (nBits_ & 7));
    ++nBits_;
  }

  void append(int num, int length) {
    for (int i = length - 1; i >= 0; --i)
      push((num >> i) & 1);
  }

  const std::vector<uint8_t>& bytes() const { return bytes_; }
  std::vector<uint8_t>& bytes() { return bytes_; }
  uint32_t nBits() const { return nBits_; }

private:
  std::vector<uint8_t> bytes_;
  uint32_t nBits_ = 0;
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITBitBuffer_H
