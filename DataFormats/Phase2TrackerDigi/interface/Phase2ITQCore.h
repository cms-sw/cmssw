#ifndef DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
#define DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
#include <vector>

class Phase2ITQCore {
  // Collects hits and creates a quarter core (16 pixel positions)

public:
  Phase2ITQCore(int rocid,
                int ccol_in,
                int qcrow_in,
                bool isneighbour_in,
                bool islast_in,
                const std::vector<int>& adcs_in,
                const std::vector<int>& hits_in);

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

  int rocid() const { return rocid_; }
  int get_col() const { return ccol_; }
  int get_row() const { return qcrow_; }

  std::vector<bool> getHitmap();
  std::vector<int> getADCs();
  std::vector<bool> encodeQCore(bool is_new_col);

  const bool operator<(const Phase2ITQCore& other) {
    if (ccol_ == other.ccol_) {
      return (ccol_ < other.ccol_);
    } else {
      return (qcrow_ < other.qcrow_);
    }
  }

private:
  std::vector<int> adcs_;  // Full array of adc values in a quarter core
  std::vector<int> hits_;  // Full array of hit occurrences
  bool islast_;            // RD53 chip encoding bits
  bool isneighbour_;       // RD53 chip encoding bits
  int rocid_;              // Chip index number
  int ccol_;               // QCore position column
  int qcrow_;              // QCore position row

  std::vector<bool> toRocCoordinates(std::vector<bool>& hitmap);
  std::vector<bool> intToBinary(int num, int length);
  bool containsHit(std::vector<bool>& hitmap);
  std::vector<bool> getHitmapCode(std::vector<bool> hitmap);
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
