#ifndef DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
#define DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
#include <array>
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITBitBuffer.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITBitReader.h"
#include <vector>
#include <cstddef>
#include <cstdint>

class Phase2ITQCore {
  // Collects hits and creates a quarter core (16 pixel positions)

public:
  Phase2ITQCore(int rocid,
                int ccolIn,
                int qcrowIn,
                bool isneighbourIn,
                bool islastIn,
                const std::vector<int>& adcsIn,
                const std::vector<int>& hitsIn);

  Phase2ITQCore() {
    rocid_ = -1;
    islast_ = false;
    isneighbour_ = false;
    ccol_ = -1;
    qcrow_ = -1;
  }

  void setIsLast(bool islast) { islast_ = islast; }
  bool islast() const { return islast_; }

  void setIsNeighbour(bool isneighbour) { isneighbour_ = isneighbour; }
  bool isneighbour() const { return isneighbour_; }

  int rocid() const { return rocid_; }
  int getCol() const { return ccol_; }
  int getRow() const { return qcrow_; }

  std::vector<bool> getHitmap();
  std::vector<int> getADCs();
  // dropTot=true skips the per-hit 4-bit ToT field entirely (binary readout mode).
  // Appends this qcore's RD53 encoding to out.
  void encodeQCore(Phase2ITBitBuffer& out, bool isNewCol, bool dropTot = false);

  bool operator<(const Phase2ITQCore& other) const {
    if (ccol_ != other.ccol_)
      return ccol_ < other.ccol_;
    return qcrow_ < other.qcrow_;
  }

  static std::vector<bool> toSensorCoordinates(const std::vector<bool>& rocHitmap);
  template <typename T>
  static std::vector<T> toRocCoordinates(const std::vector<T>& inputMap);

  // Hitmaps and ADCs are always 16 entries
  static std::array<bool, 16> decodeHitmap(Phase2ITBitReader& reader);
  static std::array<int, 16> decodeADCs(Phase2ITBitReader& reader, int numHits);

private:
  std::vector<int> adcs_;  // Full array of adc values in a quarter core
  std::vector<int> hits_;  // Full array of hit occurrences
  bool islast_;            // RD53 chip encoding bits
  bool isneighbour_;       // RD53 chip encoding bits
  int rocid_;              // Chip index number
  int ccol_;               // QCore position column
  int qcrow_;              // QCore position row

  uint32_t binaryToInt(const std::vector<bool>& binary, size_t& bitPos, int length);
  std::vector<bool> intToBinary(int num, int length);
  static bool containsHit(std::vector<bool>& hitmap);
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
