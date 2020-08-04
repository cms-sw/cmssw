#ifndef DataFormats_GEMDigi_GEMPadDigi_h
#define DataFormats_GEMDigi_GEMPadDigi_h

/** \class GEMPadDigi
 *
 * Digi for GEM trigger pads
 *
 * \author Vadim Khotilovich
 *
 */

#include "DataFormats/MuonDetId/interface/GEMSubDetId.h"

#include <cstdint>
#include <iosfwd>

class GEMPadDigi {
public:
  enum InValid { ME0InValid = 255, GE11InValid = 255, GE21InValid = 511 };
  enum NumberPartitions { ME0 = 8, GE11 = 8, GE21 = 8, GE21SplitStrip = 16 };

  explicit GEMPadDigi(uint16_t pad,
                      int16_t bx,
                      enum GEMSubDetId::Station station = GEMSubDetId::Station::GE11,
                      enum NumberPartitions nPart = NumberPartitions::GE11);
  GEMPadDigi();

  bool operator==(const GEMPadDigi& digi) const;
  bool operator!=(const GEMPadDigi& digi) const;
  bool operator<(const GEMPadDigi& digi) const;
  // only depends on the "InValid" enum so it also
  // works on unpacked data
  bool isValid() const;

  // return the pad number. counts from 0.
  uint16_t pad() const { return pad_; }
  int16_t bx() const { return bx_; }
  GEMSubDetId::Station station() const { return station_; }

  // Newer GE2/1 geometries will have 16! eta partitions
  // instead of the usual 8.
  void setNPartitions(enum NumberPartitions nPart) { part_ = nPart; }
  enum NumberPartitions nPartitions() const { return part_; }

  void print() const;

private:
  uint16_t pad_;
  int16_t bx_;
  GEMSubDetId::Station station_;
  // number of eta partitions
  enum NumberPartitions part_;
};

std::ostream& operator<<(std::ostream& o, const GEMPadDigi& digi);

#endif
