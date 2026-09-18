#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
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
  std::vector<bool> encChunk(const std::vector<bool>& chunk) {
    int n = chunk.size();
    std::vector<bool> result;
    if (n < 2)
      return result;
    std::vector<std::vector<bool>> active;
    active.push_back(chunk);
    int curSize = n;
    while (curSize > 2) {
      int half = curSize / 2;
      std::vector<std::vector<bool>> nextLevel;
      nextLevel.reserve(active.size() * 2);
      for (const auto& c : active) {
        bool lh = false, rh = false;
        for (int i = 0; i < half; ++i)
          if (c[i]) {
            lh = true;
            break;
          }
        for (int i = half; i < curSize; ++i)
          if (c[i]) {
            rh = true;
            break;
          }
        std::vector<bool> step = encPairBits(lh, rh);
        result.insert(result.end(), step.begin(), step.end());
        if (lh)
          nextLevel.emplace_back(c.begin(), c.begin() + half);
        if (rh)
          nextLevel.emplace_back(c.begin() + half, c.end());
      }
      active = std::move(nextLevel);
      curSize = half;
    }
    for (const auto& c : active) {
      std::vector<bool> m = encPairBits(c[0], c[1]);
      result.insert(result.end(), m.begin(), m.end());
    }
    return result;
  }

  // Read one 2 bits from the stream.
  std::pair<bool, bool> decPairBits(const std::vector<bool>& bits, size_t& pos) {
    if (pos >= bits.size())
      return {false, false};
    bool first = bits[pos++];
    if (!first)  // "0 -> 01" substitute
      return {false, true};
    if (pos >= bits.size())
      return {true, false};
    bool second = bits[pos++];
    if (!second)
      return {true, false};
    return {true, true};
  }

  // Decoding
  std::vector<bool> decChunk(const std::vector<bool>& bits, size_t& pos, int n) {
    std::vector<bool> result(n, false);
    std::vector<std::pair<int, int>> active;
    active.emplace_back(0, n);
    int curSize = n;
    while (curSize > 2) {
      int half = curSize / 2;
      std::vector<std::pair<int, int>> nextActive;
      nextActive.reserve(active.size() * 2);
      for (const auto& sub : active) {
        auto p = decPairBits(bits, pos);
        if (p.first)
          nextActive.emplace_back(sub.first, half);
        if (p.second)
          nextActive.emplace_back(sub.first + half, half);
      }
      active = std::move(nextActive);
      curSize = half;
    }
    for (const auto& sub : active) {
      auto p = decPairBits(bits, pos);
      result[sub.first] = p.first;
      result[sub.first + 1] = p.second;
    }
    return result;
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
  std::vector<bool> sensorHitmap(16, false);  // or HITMAP_SIZE
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

//Returns the RD53B-spec Huffman encoded hitmap.
//Layout: encPair(rowOr) || encChunk(row1) || encChunk(row2)
//  row1 = hitmap[0..7], row2 = hitmap[8..15] (ROC 2x8 layout).
std::vector<bool> Phase2ITQCore::encodeHitmap(const std::vector<bool>& hitmap) {
  std::vector<bool> code;
  if (hitmap.size() != 16)
    return code;
  std::vector<bool> row1(hitmap.begin(), hitmap.begin() + 8);
  std::vector<bool> row2(hitmap.begin() + 8, hitmap.end());
  bool row1Has = false, row2Has = false;
  for (bool b : row1)
    if (b) {
      row1Has = true;
      break;
    }
  for (bool b : row2)
    if (b) {
      row2Has = true;
      break;
    }
  std::vector<bool> rowOr = encPairBits(row1Has, row2Has);
  code.insert(code.end(), rowOr.begin(), rowOr.end());
  if (row1Has) {
    std::vector<bool> r1 = encChunk(row1);
    code.insert(code.end(), r1.begin(), r1.end());
  }
  if (row2Has) {
    std::vector<bool> r2 = encChunk(row2);
    code.insert(code.end(), r2.begin(), r2.end());
  }
  return code;
}

std::vector<bool> Phase2ITQCore::decodeHitmap(const std::vector<bool>& bitstream, size_t& bitPos) {
  std::vector<bool> hitmap(16, false);
  auto rowOr = decPairBits(bitstream, bitPos);
  if (rowOr.first) {
    std::vector<bool> row1 = decChunk(bitstream, bitPos, 8);
    for (int i = 0; i < 8; ++i)
      hitmap[i] = row1[i];
  }
  if (rowOr.second) {
    std::vector<bool> row2 = decChunk(bitstream, bitPos, 8);
    for (int i = 0; i < 8; ++i)
      hitmap[8 + i] = row2[i];
  }
  return hitmap;
}

std::vector<int> Phase2ITQCore::decodeADCs(const std::vector<bool>& bitstream, size_t& bitPos, int numHits) {
  std::vector<int> adcs;
  adcs.reserve(numHits);
  for (int i = 0; i < numHits; i++) {
    adcs.push_back(::binaryToInt(bitstream, bitPos, 4));
  }
  return adcs;
}

//Returns the bit code associated with the Phase2ITQCore
std::vector<bool> Phase2ITQCore::encodeQCore(bool isNewCol, bool dropTot) {
  std::vector<bool> code = {};

  if (isNewCol) {
    std::vector<bool> colCode = intToBinary(ccol_, 6);
    code.insert(code.end(), colCode.begin(), colCode.end());
  }

  code.push_back(islast_);
  code.push_back(isneighbour_);

  if (!isneighbour_) {
    std::vector<bool> rowCode = intToBinary(qcrow_, 8);
    code.insert(code.end(), rowCode.begin(), rowCode.end());
  }

  std::vector<bool> hitmap = getHitmap();
  std::vector<bool> hitmapCode = encodeHitmap(hitmap);
  code.insert(code.end(), hitmapCode.begin(), hitmapCode.end());

  if (!dropTot) {
    std::vector<int> adcsCode = getADCs();
    for (int i = 0; i < 16; i++) {
      if (hitmap[i]) {  // only write ADC if there's a hit
        std::vector<bool> adcCode = intToBinary(adcsCode[i], 4);
        code.insert(code.end(), adcCode.begin(), adcCode.end());
      }
    }
  }

  return code;
}
