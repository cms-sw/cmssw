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

  explicit GEMPadDigiCluster(std::vector<uint16_t> pads,
                             int16_t bx,
                             enum GEMSubDetId::Station station = GEMSubDetId::Station::GE11);
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

  void print() const;

private:
  std::vector<uint16_t> v_;
  int32_t bx_;
  GEMSubDetId::Station station_;
};

std::ostream& operator<<(std::ostream& o, const GEMPadDigiCluster& digi);

#endif
