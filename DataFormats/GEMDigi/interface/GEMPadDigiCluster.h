#ifndef DataFormats_GEMDigi_GEMPadDigiCluster_h
#define DataFormats_GEMDigi_GEMPadDigiCluster_h

/** \class GEMPadDigiCluster
 *
 * Cluster of maximal 8 adjacent GEM pad digis
 * with the same BX
 *
 * \author Sven Dildick
 *
 */

#include "DataFormats/MuonDetId/interface/GEMSubDetId.h"

#include <cstdint>
#include <iosfwd>
#include <vector>

class GEMPadDigiCluster {
public:
  enum InValid { GE11InValid = 255, GE21InValid = 511 };
  // Newer GE2/1 geometries will have 16 eta partitions
  // instead of the usual 8.
  enum NumberPartitions { ME0 = 8, GE11 = 8, GE21 = 8, GE21SplitStrip = 16 };

  explicit GEMPadDigiCluster(std::vector<uint16_t> pads,
                             int16_t bx,
                             enum GEMSubDetId::Station station = GEMSubDetId::Station::GE11,
                             unsigned nPart = NumberPartitions::GE11);
  GEMPadDigiCluster();

  bool operator==(const GEMPadDigiCluster& digi) const;
  bool operator!=(const GEMPadDigiCluster& digi) const;
  bool operator<(const GEMPadDigiCluster& digi) const;
  // only depends on the "InValid" enum so it also
  // works on unpacked data
  bool isValid() const;

  const std::vector<uint16_t>& pads() const { return v_; }
  int bx() const { return bx_; }

  GEMSubDetId::Station station() const { return station_; }

  unsigned nPartitions() const { return part_; }
  void print() const;

  int alctMatchTime() const { return alctMatchTime_; }
  void setAlctMatchTime(int matchWin) { alctMatchTime_ = matchWin; }

private:
  std::vector<uint16_t> v_;
  int32_t bx_;
  int alctMatchTime_ = -1;
  GEMSubDetId::Station station_;
  // number of eta partitions
  unsigned part_;
};

std::ostream& operator<<(std::ostream& o, const GEMPadDigiCluster& digi);

#endif
