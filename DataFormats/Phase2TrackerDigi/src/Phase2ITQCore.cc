#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"

#include <array>
#include <cmath>
#include <vector>
#include <functional>
#include <utility>

// Huffman encoding/decoding for hitmaps.
// Every split hitmaps checkeed whether "left/right has hit".
// Finally, "01 -> 0" replacing is done.
namespace {

  std::vector<bool> encPairBits(bool a, bool b) {
    if (!a && !b)
      return std::vector<bool>();
    if (!a && b)  // "01 -> 0" substitue
      return std::vector<bool>({false});
    if (a && !b)
      return std::vector<bool>({true, false});
    return std::vector<bool>({true, true});
  }

  // Encoding
  void appendPairBits(Phase2ITBitBuffer& out, bool a, bool b) {
    if (!a && !b)
      return;
    if (!a && b) {  // for "01 -> 0" substitute
      out.push(false);
      return;
    }
    out.push(true);
    out.push(b);
  }

  // MSB-first, matching Phase2ITQCore::intToBinary.
  void appendBits(Phase2ITBitBuffer& out, int num, int length) { out.append(num, length); }

  // Walk through the Huffman split tree
  void appendChunk(Phase2ITBitBuffer& out, const bool* chunk, int n) {
    if (n < 2)
      return;
    int active[16], nActive = 0;
    active[nActive++] = 0;
    int curSize = n;
    while (curSize > 2) {
      const int half = curSize / 2;
      int next[16], nNext = 0;
      for (int i = 0; i < nActive; ++i) {
        const int off = active[i];
        bool lh = false, rh = false;
        for (int k = 0; k < half; ++k)
          if (chunk[off + k]) {
            lh = true;
            break;
          }
        for (int k = half; k < curSize; ++k)
          if (chunk[off + k]) {
            rh = true;
            break;
          }
        appendPairBits(out, lh, rh);
        if (lh)
          next[nNext++] = off;
        if (rh)
          next[nNext++] = off + half;
      }
      for (int i = 0; i < nNext; ++i)
        active[i] = next[i];
      nActive = nNext;
      curSize = half;
    }
    for (int i = 0; i < nActive; ++i)
      appendPairBits(out, chunk[active[i]], chunk[active[i] + 1]);
  }

  void appendHitmapBits(Phase2ITBitBuffer& out, const std::vector<bool>& hitmap) {
    bool bits[16];
    for (int i = 0; i < 16; ++i)
      bits[i] = hitmap[i];
    bool row1Has = false, row2Has = false;
    for (int i = 0; i < 8; ++i) {
      row1Has = row1Has || bits[i];
      row2Has = row2Has || bits[8 + i];
    }
    appendPairBits(out, row1Has, row2Has);
    if (row1Has)
      appendChunk(out, bits, 8);
    if (row2Has)
      appendChunk(out, bits + 8, 8);
  }

  // Read one 2 bits from the stream.
  std::pair<bool, bool> decPairBits(Phase2ITBitReader& reader) {
    if (reader.atEnd())
      return {false, false};
    bool first = reader.next();
    if (!first)  // "0 -> 01" substitute
      return {false, true};
    if (reader.atEnd())
      return {true, false};
    bool second = reader.next();
    if (!second)
      return {true, false};
    return {true, true};
  }

  // Decoding
  void decChunk(Phase2ITBitReader& reader, int n, bool* out) {
    for (int i = 0; i < n; ++i)
      out[i] = false;
    int active[16], nActive = 0;
    active[nActive++] = 0;
    int curSize = n;
    while (curSize > 2) {
      const int half = curSize / 2;
      int next[16], nNext = 0;
      for (int i = 0; i < nActive; ++i) {
        auto p = decPairBits(reader);
        if (p.first)
          next[nNext++] = active[i];
        if (p.second)
          next[nNext++] = active[i] + half;
      }
      for (int i = 0; i < nNext; ++i)
        active[i] = next[i];
      nActive = nNext;
      curSize = half;
    }
    for (int i = 0; i < nActive; ++i) {
      auto p = decPairBits(reader);
      out[active[i]] = p.first;
      out[active[i] + 1] = p.second;
    }
  }

}  // namespace

//4x4 region of hits in sensor coordinates
Phase2ITQCore::Phase2ITQCore(int rocid,
                             int ccolIn,
                             int qcrowIn,
                             bool isneighbourIn,
                             bool islastIn,
                             const std::vector<int>& adcsIn,
                             const std::vector<int>& hitsIn) {
  rocid_ = rocid;
  ccol_ = ccolIn;
  qcrow_ = qcrowIn;
  isneighbour_ = isneighbourIn;
  islast_ = islastIn;
  adcs_ = adcsIn;
  hits_ = hitsIn;
}

//Takes a hitmap in sensor coordinates in 4x4 and converts it to readout chip coordinates with 2x8
template <typename T>
std::vector<T> Phase2ITQCore::toRocCoordinates(const std::vector<T>& inputMap) {
  std::vector<T> rocCoord(16);

  for (size_t i = 0; i < inputMap.size(); i++) {
    int row = i / 4;
    int col = i % 4;
    int newRow;
    int newCol;

    if (row % 2 == 0) {
      newRow = row / 2;
      newCol = 2 * col;
    } else {
      newRow = row / 2;
      newCol = 2 * col + 1;
    }

    int newIndex = 8 * newRow + newCol;
    rocCoord[newIndex] = inputMap[i];
  }

  return rocCoord;
}

std::vector<bool> Phase2ITQCore::toSensorCoordinates(const std::vector<bool>& rocHitmap) {
  std::vector<bool> sensorHitmap(16, false);  // FIXME or HITMAP_SIZE?
  for (int i = 0; i < 16; ++i) {
    int rocRow = i / 8;
    int rocCol = i % 8;
    int sensorRow = (rocCol % 2 == 0) ? rocRow * 2 : rocRow * 2 + 1;
    int sensorCol = (rocCol % 2 == 0) ? rocCol / 2 : (rocCol - 1) / 2;
    int sensorIndex = sensorRow * 4 + sensorCol;
    sensorHitmap[sensorIndex] = rocHitmap[i];
  }
  return sensorHitmap;
}

//Returns the hitmap for the Phase2ITQCore in 4x4 sensor coordinates
std::vector<bool> Phase2ITQCore::getHitmap() {
  std::vector<bool> hitmap = {};

  hitmap.reserve(hits_.size());
  for (auto hit : hits_) {
    hitmap.push_back(hit > 0);
  }

  return (toRocCoordinates(hitmap));
}

std::vector<int> Phase2ITQCore::getADCs() {
  std::vector<int> adcmap = {};

  adcmap.reserve(adcs_.size());
  for (auto adc : adcs_) {
    adcmap.push_back(adc);
  }

  return (toRocCoordinates(adcmap));
}

//Converts an integer into a binary, and formats it with the given length
std::vector<bool> Phase2ITQCore::intToBinary(int num, int length) {
  std::vector<bool> biNum(length, false);

  for (int i = 0; i < length; ++i) {
    // Extract the (length - 1 - i)th bit from num
    biNum[i] = (num >> (length - 1 - i)) & 1;
  }

  return biNum;
}

static uint32_t binaryToInt(const std::vector<bool>& binary, size_t& bitPos, int length) {
  uint32_t result = 0;
  for (int i = 0; i < length; ++i) {
    if ((bitPos + i) < binary.size() && binary[bitPos + i]) {
      result |= (1 << (length - 1 - i));
    }
  }
  bitPos += length;
  return result;
}

//Takes a hitmap and returns true if it contains any hits
bool Phase2ITQCore::containsHit(std::vector<bool>& hitmap) {
  bool foundHit = false;
  for (size_t i = 0; i < hitmap.size(); i++) {
    if (hitmap[i]) {
      foundHit = true;
      break;
    }
  }

  return foundHit;
}

std::array<bool, 16> Phase2ITQCore::decodeHitmap(Phase2ITBitReader& reader) {
  std::array<bool, 16> hitmap{};
  auto rowOr = decPairBits(reader);
  if (rowOr.first)
    decChunk(reader, 8, hitmap.data());
  if (rowOr.second)
    decChunk(reader, 8, hitmap.data() + 8);
  return hitmap;
}

std::array<int, 16> Phase2ITQCore::decodeADCs(Phase2ITBitReader& reader, int numHits) {
  std::array<int, 16> adcs{};
  for (int i = 0; i < numHits && i < 16; i++) {
    adcs[i] = reader.bits(4);
  }
  return adcs;
}

//Returns the bit code associated with the Phase2ITQCore
void Phase2ITQCore::encodeQCore(Phase2ITBitBuffer& code, bool isNewCol, bool dropTot) {
  if (isNewCol) {
    appendBits(code, ccol_, 6);
  }

  code.push(islast_);
  code.push(isneighbour_);

  if (!isneighbour_) {
    appendBits(code, qcrow_, 8);
  }

  std::vector<bool> hitmap = getHitmap();
  appendHitmapBits(code, hitmap);

  if (!dropTot) {
    std::vector<int> adcsCode = getADCs();
    for (int i = 0; i < 16; i++) {
      if (hitmap[i]) {  // write ADC only if there's a hit
        appendBits(code, adcsCode[i], 4);
      }
    }
  }
}
