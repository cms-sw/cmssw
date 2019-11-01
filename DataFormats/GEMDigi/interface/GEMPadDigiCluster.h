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

#include <cstdint>
#include <iosfwd>
#include <vector>

class GEMPadDigiCluster {
public:
  explicit GEMPadDigiCluster(std::vector<uint16_t> pads, int bx);
  GEMPadDigiCluster();

  bool operator==(const GEMPadDigiCluster& digi) const;
  bool operator!=(const GEMPadDigiCluster& digi) const;
  bool operator<(const GEMPadDigiCluster& digi) const;
  bool isValid() const;

  const std::vector<uint16_t>& pads() const { return v_; }
  int bx() const { return bx_; }

  void print() const;

private:
  std::vector<uint16_t> v_;
  int32_t bx_;
};

std::ostream& operator<<(std::ostream& o, const GEMPadDigiCluster& digi);

#endif
