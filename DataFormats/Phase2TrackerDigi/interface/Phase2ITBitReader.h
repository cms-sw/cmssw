#ifndef DataFormats_Phase2TrackerDigi_Phase2ITBitReader_H
#define DataFormats_Phase2TrackerDigi_Phase2ITBitReader_H

#include <cstdint>

// Just for reading of the bytes, owning them as it is being read
class Phase2ITBitReader {
public:
  Phase2ITBitReader(const uint8_t* bytes, uint32_t nBits) : bytes_(bytes), nBits_(nBits) {}

  bool atEnd() const { return pos_ >= nBits_; }
  uint32_t pos() const { return pos_; }

  bool next() {
    if (pos_ >= nBits_)
      return false;
    const bool b = (bytes_[pos_ / 8] >> (7 - pos_ % 8)) & 1;
    ++pos_;
    return b;
  }

  uint32_t bits(int length) {
    uint32_t result = 0;
    for (int i = 0; i < length && pos_ < nBits_; i++)
      result = (result << 1) | (next() ? 1u : 0u);
    return result;
  }

private:
  const uint8_t* bytes_;
  uint32_t nBits_;
  uint32_t pos_ = 0;
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITBitReader_H
