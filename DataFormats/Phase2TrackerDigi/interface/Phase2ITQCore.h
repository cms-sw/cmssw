#ifndef DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
#define DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
#include <vector>

class Phase2ITQCore {
  // Collects hits and creates a quarter core (16 pixel positions)
private:
  std::vector<int> adcs;  // Full array of adc values in a quarter core
  std::vector<int> hits;  // Full array of hit occurrences
  bool islast_;           // RD53 chip encoding bits
  bool isneighbour_;      // RD53 chip encoding bits
  int rocid_;             // Chip index number
  int ccol;               // QCore position column
  int qcrow;              // QCore position row

public:
  Phase2ITQCore(int rocid,
                int ccol_in,
                int qcrow_in,
                bool isneighbour_in,
                bool islast_in,
                std::vector<int> adcs_in,
                std::vector<int> hits_in);

  Phase2ITQCore() {
    rocid_ = -1;
    islast_ = false;
    isneighbour_ = false;
    ccol = -1;
    qcrow = -1;
  }

  void setIsLast(bool islast) { islast_ = islast; }
  bool islast() const { return islast_; }

  void setIsNeighbour(bool isneighbour) { isneighbour_ = isneighbour; }

  int rocid() const { return rocid_; }
  int get_col() const { return ccol; }
  int get_row() const { return qcrow; }

  std::vector<bool> getHitmap();
  std::vector<int> getADCs();
  std::vector<bool> encodeQCore(bool is_new_col);

  const bool operator<(const Phase2ITQCore& other) {
    if (ccol == other.ccol) {
      return (ccol < other.ccol);
    } else {
      return (qcrow < other.qcrow);
    }
  }

private:
  std::vector<bool> toRocCoordinates(std::vector<bool>& hitmap);
  std::vector<bool> intToBinary(int num, int length);
  bool containsHit(std::vector<bool>& hitmap);
  std::vector<bool> getHitmapCode(std::vector<bool> hitmap);
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITQCore_H
